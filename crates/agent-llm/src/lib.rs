use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::env;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ModelId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for ModelId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProviderId(String);

impl ProviderId {
    pub const OPENAI: &'static str = "openai";
    pub const OPENAI_COMPATIBLE: &'static str = "openai_compatible";
    pub const OPENAI_RESPONSES_COMPATIBLE: &'static str = "openai_responses_compatible";
    pub const ANTHROPIC: &'static str = "anthropic";
    pub const GOOGLE: &'static str = "google";
    pub const OPENCODE: &'static str = "opencode";
    pub const CODEX: &'static str = "codex";

    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ProviderId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for ProviderId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    Llm,
    ContextEngine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderResponse {
    OpenAi,
    OpenAiResponses,
    Anthropic,
    Bedrock,
    Google,
    OpenCode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum AuthMethod {
    None,
    ApiKeyEnv { env: String },
    OAuth { provider: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: ModelId,
    pub display_name: Option<String>,
    pub context_window: Option<u32>,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
}

impl Model {
    pub fn new(id: impl Into<ModelId>) -> Self {
        Self {
            id: id.into(),
            display_name: None,
            context_window: None,
            supports_tools: true,
            supports_reasoning: false,
        }
    }

    pub fn reasoning(mut self, enabled: bool) -> Self {
        self.supports_reasoning = enabled;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    pub id: ProviderId,
    pub provider_type: ProviderType,
    pub response: ProviderResponse,
    pub base_url: String,
    pub models: Vec<Model>,
    pub auth_methods: Vec<AuthMethod>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
}

impl Provider {
    pub fn new(
        id: impl Into<ProviderId>,
        response: ProviderResponse,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            provider_type: ProviderType::Llm,
            response,
            base_url: base_url.into(),
            models: Vec::new(),
            auth_methods: Vec::new(),
            headers: BTreeMap::new(),
        }
    }

    pub fn with_models(mut self, models: Vec<Model>) -> Self {
        self.models = models;
        self
    }

    pub fn with_auth(mut self, auth_methods: Vec<AuthMethod>) -> Self {
        self.auth_methods = auth_methods;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: ModelId,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub options: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub text: String,
    pub raw: Value,
    pub route: BackendRoute,
}

impl ChatRequest {
    pub fn user(model: impl Into<ModelId>, content: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: vec![ChatMessage::user(content)],
            tools: Vec::new(),
            temperature: None,
            max_output_tokens: None,
            reasoning_effort: None,
            options: Map::new(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("provider not found: {0}")]
    ProviderNotFound(String),
    #[error("unsupported provider response type: {0:?}")]
    UnsupportedResponse(ProviderResponse),
    #[error("missing API key env var: {0}")]
    MissingApiKey(String),
    #[error("missing provider URL env var: {0}")]
    MissingUrlEnv(String),
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("provider returned HTTP {status}: {body}")]
    HttpStatus { status: u16, body: String },
    #[error("response did not include text output")]
    MissingOutputText,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    MessageDelta { delta: String },
    ToolCallRequested { name: String, arguments: Value },
    Completed { usage: Option<Value> },
    Error { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendRoute {
    pub response: ProviderResponse,
    pub endpoint: String,
}

#[derive(Debug, Default, Clone)]
pub struct ProviderRouter;

impl ProviderRouter {
    pub fn route(&self, provider: &Provider, model_id: &ModelId) -> BackendRoute {
        if provider.response == ProviderResponse::OpenCode {
            return route_opencode(provider, model_id);
        }

        if provider.response == ProviderResponse::OpenAi
            && model_id.as_str().contains("gpt-5")
            && matches!(provider.id.as_str(), ProviderId::OPENAI | ProviderId::CODEX)
        {
            return BackendRoute {
                response: ProviderResponse::OpenAiResponses,
                endpoint: responses_endpoint(&provider.base_url),
            };
        }

        BackendRoute {
            response: provider.response,
            endpoint: provider.base_url.clone(),
        }
    }
}

pub trait RequestTransform: Send + Sync {
    fn name(&self) -> &'static str;
    fn transform(&self, provider: &Provider, request: ChatRequest) -> ChatRequest;
}

#[derive(Default)]
pub struct TransformPipeline {
    transforms: Vec<Box<dyn RequestTransform>>,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn pipe<T>(mut self, transform: T) -> Self
    where
        T: RequestTransform + 'static,
    {
        self.transforms.push(Box::new(transform));
        self
    }

    pub fn transform(&self, provider: &Provider, mut request: ChatRequest) -> ChatRequest {
        for transform in &self.transforms {
            request = transform.transform(provider, request);
        }
        request
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.transforms
            .iter()
            .map(|transform| transform.name())
            .collect()
    }
}

pub struct MergeSystemMessages;

impl RequestTransform for MergeSystemMessages {
    fn name(&self) -> &'static str {
        "merge_system_messages"
    }

    fn transform(&self, _provider: &Provider, mut request: ChatRequest) -> ChatRequest {
        let mut system = Vec::new();
        let mut rest = Vec::new();
        for message in request.messages {
            if message.role == ChatRole::System {
                system.push(message.content);
            } else {
                rest.push(message);
            }
        }
        if !system.is_empty() {
            let mut messages = vec![ChatMessage::system(system.join("\n\n"))];
            messages.extend(rest);
            request.messages = messages;
        } else {
            request.messages = rest;
        }
        request
    }
}

pub struct CodexResponsesCompat;

impl RequestTransform for CodexResponsesCompat {
    fn name(&self) -> &'static str {
        "codex_responses_compat"
    }

    fn transform(&self, provider: &Provider, mut request: ChatRequest) -> ChatRequest {
        if provider.id.as_str() != ProviderId::CODEX {
            return request;
        }

        request.temperature = None;
        request.max_output_tokens = None;
        request
            .options
            .insert("store".to_string(), Value::Bool(false));
        request.options.insert(
            "include".to_string(),
            Value::Array(vec![Value::String(
                "reasoning.encrypted_content".to_string(),
            )]),
        );
        request
    }
}

pub fn default_pipeline() -> TransformPipeline {
    TransformPipeline::new()
        .pipe(MergeSystemMessages)
        .pipe(CodexResponsesCompat)
}

pub fn built_in_providers() -> Vec<Provider> {
    vec![
        Provider::new(
            ProviderId::OPENAI,
            ProviderResponse::OpenAi,
            "https://api.openai.com/v1/chat/completions",
        )
        .with_models(vec![
            Model::new("gpt-5.1").reasoning(true),
            Model::new("gpt-5.1-codex").reasoning(true),
        ])
        .with_auth(vec![AuthMethod::ApiKeyEnv {
            env: "OPENAI_API_KEY".to_string(),
        }]),
        Provider::new(
            ProviderId::OPENAI_COMPATIBLE,
            ProviderResponse::OpenAi,
            "{{OPENAI_BASE_URL}}/v1/chat/completions",
        )
        .with_auth(vec![AuthMethod::ApiKeyEnv {
            env: "OPENAI_API_KEY".to_string(),
        }]),
        Provider::new(
            ProviderId::OPENAI_RESPONSES_COMPATIBLE,
            ProviderResponse::OpenAiResponses,
            "{{OPENAI_BASE_URL}}/v1/responses",
        )
        .with_auth(vec![AuthMethod::ApiKeyEnv {
            env: "OPENAI_API_KEY".to_string(),
        }]),
        Provider::new(
            ProviderId::ANTHROPIC,
            ProviderResponse::Anthropic,
            "https://api.anthropic.com/v1/messages",
        )
        .with_models(vec![Model::new("claude-sonnet-4.5")])
        .with_auth(vec![AuthMethod::ApiKeyEnv {
            env: "ANTHROPIC_API_KEY".to_string(),
        }]),
        Provider::new(
            ProviderId::GOOGLE,
            ProviderResponse::Google,
            "https://generativelanguage.googleapis.com/v1beta",
        )
        .with_models(vec![Model::new("gemini-3-pro")])
        .with_auth(vec![AuthMethod::ApiKeyEnv {
            env: "GOOGLE_API_KEY".to_string(),
        }]),
        Provider::new(
            ProviderId::OPENCODE,
            ProviderResponse::OpenCode,
            "{{OPENCODE_BASE_URL}}",
        )
        .with_models(vec![
            Model::new("claude-sonnet-4.5"),
            Model::new("gpt-5.1-codex").reasoning(true),
            Model::new("gemini-3-pro"),
        ])
        .with_auth(vec![AuthMethod::ApiKeyEnv {
            env: "OPENCODE_API_KEY".to_string(),
        }]),
    ]
}

pub fn find_provider(id: &str) -> Option<Provider> {
    built_in_providers()
        .into_iter()
        .find(|provider| provider.id.as_str() == id)
}

pub struct ProviderClient {
    router: ProviderRouter,
    pipeline: TransformPipeline,
    http: reqwest::blocking::Client,
}

impl ProviderClient {
    pub fn new() -> Self {
        Self {
            router: ProviderRouter,
            pipeline: default_pipeline(),
            http: reqwest::blocking::Client::new(),
        }
    }

    pub fn chat(&self, provider: Provider, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let provider = resolve_provider_templates(provider)?;
        let request = self.pipeline.transform(&provider, request);
        let route = self.router.route(&provider, &request.model);

        match route.response {
            ProviderResponse::OpenAiResponses => self.openai_responses(provider, route, request),
            response => Err(LlmError::UnsupportedResponse(response)),
        }
    }

    fn openai_responses(
        &self,
        provider: Provider,
        route: BackendRoute,
        request: ChatRequest,
    ) -> Result<ChatResponse, LlmError> {
        let api_key = provider_api_key(&provider)?;
        let body = responses_body(&request);
        let response = self
            .http
            .post(&route.endpoint)
            .bearer_auth(api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()?;
        let status = response.status();
        let raw: Value = response.json()?;

        if !status.is_success() {
            return Err(LlmError::HttpStatus {
                status: status.as_u16(),
                body: raw.to_string(),
            });
        }

        let text = extract_response_text(&raw).ok_or(LlmError::MissingOutputText)?;
        Ok(ChatResponse { text, raw, route })
    }
}

impl Default for ProviderClient {
    fn default() -> Self {
        Self::new()
    }
}

pub fn chat_once(
    provider_id: &str,
    model: impl Into<ModelId>,
    prompt: impl Into<String>,
) -> Result<ChatResponse, LlmError> {
    let provider = find_provider(provider_id)
        .ok_or_else(|| LlmError::ProviderNotFound(provider_id.to_string()))?;
    ProviderClient::new().chat(provider, ChatRequest::user(model, prompt))
}

pub fn resolve_provider_templates(mut provider: Provider) -> Result<Provider, LlmError> {
    provider.base_url = render_env_template(&provider.base_url)?;
    for value in provider.headers.values_mut() {
        *value = render_env_template(value)?;
    }
    Ok(provider)
}

pub fn responses_body(request: &ChatRequest) -> Value {
    let mut body = Map::new();
    body.insert(
        "model".to_string(),
        Value::String(request.model.as_str().to_string()),
    );
    body.insert("input".to_string(), responses_input(&request.messages));

    if let Some(max_output_tokens) = request.max_output_tokens {
        body.insert(
            "max_output_tokens".to_string(),
            Value::Number(max_output_tokens.into()),
        );
    }
    if let Some(temperature) = request.temperature {
        body.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(effort) = &request.reasoning_effort {
        body.insert("reasoning".to_string(), json!({ "effort": effort }));
    }

    for (key, value) in &request.options {
        body.insert(key.clone(), value.clone());
    }

    Value::Object(body)
}

fn responses_input(messages: &[ChatMessage]) -> Value {
    Value::Array(
        messages
            .iter()
            .map(|message| {
                json!({
                    "role": responses_role(&message.role),
                    "content": message.content,
                })
            })
            .collect(),
    )
}

fn responses_role(role: &ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "user",
    }
}

pub fn extract_response_text(raw: &Value) -> Option<String> {
    if let Some(text) = raw.get("output_text").and_then(Value::as_str) {
        if !text.is_empty() {
            return Some(text.to_string());
        }
    }

    let mut chunks = Vec::new();
    for item in raw.get("output")?.as_array()? {
        let Some(content) = item.get("content").and_then(Value::as_array) else {
            continue;
        };
        for part in content {
            let kind = part.get("type").and_then(Value::as_str);
            if matches!(kind, Some("output_text" | "text"))
                && let Some(text) = part.get("text").and_then(Value::as_str)
            {
                chunks.push(text.to_string());
            }
        }
    }

    if chunks.is_empty() {
        None
    } else {
        Some(chunks.join(""))
    }
}

fn provider_api_key(provider: &Provider) -> Result<String, LlmError> {
    for method in &provider.auth_methods {
        if let AuthMethod::ApiKeyEnv { env: env_name } = method {
            let value =
                env::var(env_name).map_err(|_| LlmError::MissingApiKey(env_name.clone()))?;
            if !value.trim().is_empty() {
                return Ok(value);
            }
        }
    }
    Err(LlmError::MissingApiKey("OPENAI_API_KEY".to_string()))
}

fn render_env_template(input: &str) -> Result<String, LlmError> {
    let mut output = String::new();
    let mut rest = input;

    while let Some(start) = rest.find("{{") {
        let (before, after_start) = rest.split_at(start);
        output.push_str(before);
        let after_start = &after_start[2..];
        let Some(end) = after_start.find("}}") else {
            output.push_str("{{");
            output.push_str(after_start);
            return Ok(output);
        };
        let name = after_start[..end].trim();
        let value = env::var(name).map_err(|_| LlmError::MissingUrlEnv(name.to_string()))?;
        output.push_str(value.trim_end_matches('/'));
        rest = &after_start[end + 2..];
    }

    output.push_str(rest);
    Ok(output)
}

fn route_opencode(provider: &Provider, model_id: &ModelId) -> BackendRoute {
    let model = model_id.as_str();
    let (response, suffix) = if model.starts_with("claude-") {
        (ProviderResponse::Anthropic, "/v1/messages")
    } else if model.starts_with("gpt-5") {
        (ProviderResponse::OpenAiResponses, "/v1/responses")
    } else if model.starts_with("gemini-") {
        (ProviderResponse::Google, "/v1")
    } else {
        (ProviderResponse::OpenAi, "/v1/chat/completions")
    };

    BackendRoute {
        response,
        endpoint: append_endpoint(&provider.base_url, suffix),
    }
}

fn append_endpoint(base_url: &str, suffix: &str) -> String {
    format!("{}{}", base_url.trim_end_matches('/'), suffix)
}

fn responses_endpoint(base_url: &str) -> String {
    let base_url = base_url.trim_end_matches('/');
    if let Some(prefix) = base_url.strip_suffix("/chat/completions") {
        return format!("{prefix}/responses");
    }
    if base_url.ends_with("/responses") {
        return base_url.to_string();
    }
    append_endpoint(base_url, "/v1/responses")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_openai_gpt5_to_responses() {
        let provider = find_provider(ProviderId::OPENAI).unwrap();
        let route = ProviderRouter.route(&provider, &ModelId::from("gpt-5.1"));
        assert_eq!(route.response, ProviderResponse::OpenAiResponses);
        assert_eq!(route.endpoint, "https://api.openai.com/v1/responses");
    }

    #[test]
    fn routes_opencode_by_model_prefix() {
        let provider = find_provider(ProviderId::OPENCODE).unwrap();
        let router = ProviderRouter;

        assert_eq!(
            router
                .route(&provider, &ModelId::from("claude-sonnet-4.5"))
                .response,
            ProviderResponse::Anthropic
        );
        assert_eq!(
            router
                .route(&provider, &ModelId::from("gpt-5.1-codex"))
                .response,
            ProviderResponse::OpenAiResponses
        );
        assert_eq!(
            router
                .route(&provider, &ModelId::from("gemini-3-pro"))
                .response,
            ProviderResponse::Google
        );
    }

    #[test]
    fn codex_transform_strips_unsupported_fields() {
        let provider = Provider::new(
            ProviderId::CODEX,
            ProviderResponse::OpenAiResponses,
            "unused",
        );
        let mut request = ChatRequest::user("gpt-5.1-codex", "hello");
        request.temperature = Some(0.7);
        request.max_output_tokens = Some(1000);

        let request = default_pipeline().transform(&provider, request);

        assert_eq!(request.temperature, None);
        assert_eq!(request.max_output_tokens, None);
        assert_eq!(request.options.get("store"), Some(&Value::Bool(false)));
    }

    #[test]
    fn builds_responses_body() {
        let mut request = ChatRequest::user("gpt-5.1", "hello");
        request.max_output_tokens = Some(128);
        request.reasoning_effort = Some("low".to_string());
        let body = responses_body(&request);

        assert_eq!(body["model"], "gpt-5.1");
        assert_eq!(body["input"][0]["role"], "user");
        assert_eq!(body["input"][0]["content"], "hello");
        assert_eq!(body["max_output_tokens"], 128);
        assert_eq!(body["reasoning"]["effort"], "low");
    }

    #[test]
    fn extracts_nested_response_text() {
        let raw = json!({
            "output": [{
                "type": "message",
                "content": [
                    { "type": "output_text", "text": "hello" },
                    { "type": "output_text", "text": " world" }
                ]
            }]
        });

        assert_eq!(extract_response_text(&raw), Some("hello world".to_string()));
    }

    #[test]
    fn renders_env_template() {
        // SAFETY: this process env var is namespaced for this unit test.
        unsafe {
            env::set_var("SEED_AGENT_TEST_BASE_URL", "https://example.test/");
        }
        let rendered = render_env_template("{{SEED_AGENT_TEST_BASE_URL}}/v1/responses").unwrap();
        assert_eq!(rendered, "https://example.test/v1/responses");
    }
}
