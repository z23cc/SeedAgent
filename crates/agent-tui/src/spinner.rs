use std::io::{IsTerminal, Write, stderr};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const TICK_MS: u64 = 80;
const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Clone)]
struct SpinnerLine {
    label: String,
    subtitle: Option<String>,
    /// Wall-clock when the current label was set. Used to auto-paint
    /// "(Ns since label)" once a tool stays under the same label for >5s
    /// without explicit subtitle updates — gives long-running tools (like
    /// plan_verify firing the RepoPrompt verifier) visible progress.
    label_started: Instant,
}

impl Default for SpinnerLine {
    fn default() -> Self {
        Self {
            label: String::new(),
            subtitle: None,
            label_started: Instant::now(),
        }
    }
}

enum Msg {
    SetLabel(String),
    SetSubtitle(Option<String>),
    Pause,
    Resume,
    Stop,
}

pub struct Spinner {
    tx: Option<Sender<Msg>>,
    handle: Option<JoinHandle<()>>,
    started: Instant,
    enabled: bool,
    paused_flag: Arc<AtomicBool>,
}

impl Spinner {
    pub fn start(label: impl Into<String>) -> Self {
        let enabled = stderr().is_terminal();
        let started = Instant::now();
        let paused_flag = Arc::new(AtomicBool::new(false));

        if !enabled {
            return Self {
                tx: None,
                handle: None,
                started,
                enabled,
                paused_flag,
            };
        }

        let line = Arc::new(Mutex::new(SpinnerLine {
            label: label.into(),
            subtitle: None,
            label_started: Instant::now(),
        }));
        let paused_thread = paused_flag.clone();
        let line_thread = line.clone();
        let started_thread = started;

        let (tx, rx) = channel::<Msg>();
        let handle = thread::spawn(move || {
            let mut tick: u128 = 0;
            loop {
                while let Ok(msg) = rx.try_recv() {
                    match msg {
                        Msg::SetLabel(label) => {
                            if let Ok(mut guard) = line_thread.lock() {
                                guard.label = label;
                                guard.label_started = Instant::now();
                                // Reset stale subtitle so the new label starts
                                // clean rather than carrying the prior tool's
                                // chars-streaming or elapsed annotation.
                                guard.subtitle = None;
                            }
                        }
                        Msg::SetSubtitle(sub) => {
                            if let Ok(mut guard) = line_thread.lock() {
                                guard.subtitle = sub;
                            }
                        }
                        Msg::Pause => {
                            paused_thread.store(true, Ordering::Release);
                            clear_line();
                        }
                        Msg::Resume => paused_thread.store(false, Ordering::Release),
                        Msg::Stop => {
                            clear_line();
                            return;
                        }
                    }
                }

                if !paused_thread.load(Ordering::Acquire) {
                    let snapshot = line_thread
                        .lock()
                        .map(|guard| guard.clone())
                        .unwrap_or_default();
                    let frame = FRAMES[(tick as usize) % FRAMES.len()];
                    let elapsed = format_elapsed(started_thread.elapsed());
                    let label_elapsed = snapshot.label_started.elapsed();
                    // Auto-subtitle: once the same label has been live for >5s
                    // without an explicit subtitle update, surface its own
                    // elapsed so the operator can tell a slow tool from a
                    // stuck one. Falls back to the explicit subtitle when set.
                    let effective_subtitle = match snapshot.subtitle.as_deref() {
                        Some(sub) if !sub.is_empty() => Some(sub.to_string()),
                        _ if label_elapsed.as_secs() >= 5 => {
                            Some(format!("in tool {}", format_elapsed(label_elapsed)))
                        }
                        _ => None,
                    };
                    let body = match effective_subtitle {
                        Some(sub) => format!("{frame} {} · {elapsed} · {sub}", snapshot.label),
                        None => format!("{frame} {} · {elapsed}", snapshot.label),
                    };
                    paint(&body);
                }
                tick += 1;
                thread::sleep(Duration::from_millis(TICK_MS));
            }
        });

        Self {
            tx: Some(tx),
            handle: Some(handle),
            started,
            enabled,
            paused_flag,
        }
    }

    pub fn set_label(&self, label: impl Into<String>) {
        self.send(Msg::SetLabel(label.into()));
    }

    pub fn set_subtitle(&self, subtitle: Option<String>) {
        self.send(Msg::SetSubtitle(subtitle));
    }

    pub fn pause(&self) {
        if !self.enabled {
            return;
        }
        self.send(Msg::Pause);
        // Block until the spinner thread acknowledges the pause by flipping the
        // shared flag. Without this, the caller can race against an in-flight
        // paint and end up with the spinner frame concatenated onto the next
        // line of output (e.g. `14.7sseed → ...`).
        let deadline = Instant::now() + Duration::from_millis(200);
        while Instant::now() < deadline {
            if self.paused_flag.load(Ordering::Acquire) {
                break;
            }
            thread::sleep(Duration::from_millis(2));
        }
        // Force one final clear in the caller's thread so any frame painted
        // between send and processing is wiped before the caller writes.
        clear_line();
    }

    pub fn resume(&self) {
        self.send(Msg::Resume);
    }

    pub fn stop(mut self) {
        self.shutdown();
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    fn shutdown(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Msg::Stop);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.paused_flag.store(false, Ordering::Release);
    }

    fn send(&self, msg: Msg) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(msg);
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn paint(body: &str) {
    let mut err = stderr().lock();
    let _ = write!(err, "\r\x1b[2K{body}");
    let _ = err.flush();
}

fn clear_line() {
    let mut err = stderr().lock();
    let _ = write!(err, "\r\x1b[2K");
    let _ = err.flush();
}

pub fn format_elapsed(elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64();
    if secs < 1.0 {
        format!("{}ms", elapsed.as_millis())
    } else if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        let mins = (secs / 60.0).floor() as u64;
        let rem = secs - (mins as f64) * 60.0;
        format!("{mins}m{rem:.0}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_elapsed_units_match_magnitude() {
        assert_eq!(format_elapsed(Duration::from_millis(250)), "250ms");
        assert_eq!(format_elapsed(Duration::from_millis(1240)), "1.2s");
        assert_eq!(
            format_elapsed(Duration::from_secs(65) + Duration::from_millis(300)),
            "1m5s"
        );
    }

    #[test]
    fn spinner_lifecycle_does_not_panic_off_tty() {
        let spinner = Spinner::start("test");
        spinner.set_label("test2");
        spinner.set_subtitle(Some("sub".to_string()));
        spinner.set_subtitle(None);
        spinner.pause();
        spinner.resume();
        spinner.stop();
    }
}
