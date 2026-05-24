//! `seed replay`: dump a session's JSONL events with timestamps.

use agent_core::session::SessionStore;
use anyhow::Result;

pub(crate) fn replay(store: &SessionStore, session: Option<&str>) -> Result<()> {
    let records = store.read(session)?;
    for record in records {
        println!(
            "{} {}",
            record.ts.format("%Y-%m-%d %H:%M:%S"),
            serde_json::to_string(&record.event)?
        );
    }
    Ok(())
}
