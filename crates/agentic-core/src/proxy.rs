use std::pin::Pin;
use std::time::Duration;

use async_stream::stream;
use bytes::Bytes;
use futures::{Stream, StreamExt, TryStreamExt};
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use reqwest::Client;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use tracing::{debug, warn};

use crate::config::{Config, resolve_model_alias};
use crate::error::Error;
use crate::tool::{
    alternate_model_visible_namespace_member_name, legacy_model_visible_namespace_member_name,
    model_visible_namespace_member_name,
};

const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
];

const REQUEST_DROP_EXTRA: &[&str] = &["host", "content-length"];

fn is_hop_by_hop(name: &str) -> bool {
    HOP_BY_HOP.iter().any(|h| h.eq_ignore_ascii_case(name))
}

fn is_request_drop(name: &str) -> bool {
    is_hop_by_hop(name) || REQUEST_DROP_EXTRA.iter().any(|h| h.eq_ignore_ascii_case(name))
}

pub struct ProxyRequest {
    pub headers: HeaderMap,
    pub body: Bytes,
    pub query: Option<String>,
}

pub enum ProxyBody {
    Full(Bytes),
    Stream(Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>),
}

#[derive(Clone, Debug)]
struct NamespaceMemberName {
    namespace: String,
    name: String,
}

#[derive(Clone, Debug)]
struct NamespaceCallMapping {
    member: NamespaceMemberName,
    upstream_name: String,
    strip_container_arguments: bool,
}

#[derive(Clone, Debug, Default)]
struct NamespaceNormalization {
    calls: HashMap<String, NamespaceCallMapping>,
    original_tools: Option<Value>,
}

impl NamespaceNormalization {
    fn is_empty(&self) -> bool {
        self.calls.is_empty() && self.original_tools.is_none()
    }
}

struct NormalizedProxyRequest {
    body: Bytes,
    namespace: NamespaceNormalization,
}

pub struct ProxyResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: ProxyBody,
}

#[derive(Clone)]
pub struct ProxyState {
    pub config: Config,
    pub stream_client: Client,
    pub non_stream_client: Client,
}

impl ProxyState {
    /// # Errors
    ///
    /// Returns an error if the HTTP clients cannot be built.
    pub fn new(config: Config) -> Result<Self, Error> {
        let stream_client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(900))
            .pool_max_idle_per_host(0)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(Error::HttpClient)?;

        let non_stream_client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(300))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(Error::HttpClient)?;

        Ok(Self {
            config,
            stream_client,
            non_stream_client,
        })
    }
}

fn filter_request_headers(headers: &HeaderMap, config: &Config) -> reqwest::header::HeaderMap {
    let mut out = reqwest::header::HeaderMap::new();
    for (name, value) in headers {
        if is_request_drop(name.as_str()) {
            continue;
        }
        if let Ok(n) = reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes()) {
            if let Ok(v) = reqwest::header::HeaderValue::from_bytes(value.as_bytes()) {
                out.append(n, v);
            }
        }
    }

    let has_auth = out.contains_key(reqwest::header::AUTHORIZATION);
    if !has_auth {
        if let Some(key) = config.openai_api_key.as_deref() {
            let trimmed = key.trim();
            if !trimmed.is_empty() {
                if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {trimmed}")) {
                    out.insert(reqwest::header::AUTHORIZATION, v);
                }
            }
        }
    }

    out
}

fn filter_response_headers(headers: &reqwest::header::HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in headers {
        if is_hop_by_hop(name.as_str()) {
            continue;
        }
        if let Ok(n) = HeaderName::from_bytes(name.as_str().as_bytes()) {
            if let Ok(v) = HeaderValue::from_bytes(value.as_bytes()) {
                out.append(n, v);
            }
        }
    }
    out
}

fn is_sse_content_type(headers: &reqwest::header::HeaderMap) -> bool {
    headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.to_ascii_lowercase().starts_with("text/event-stream"))
}

fn normalize_developer_roles(value: &mut Value) -> bool {
    match value {
        Value::Object(object) => {
            let mut changed = false;
            if object.get("role").and_then(Value::as_str) == Some("developer") {
                object.insert("role".to_string(), Value::String("system".to_string()));
                changed = true;
            }
            for item in object.values_mut() {
                changed |= normalize_developer_roles(item);
            }
            changed
        }
        Value::Array(items) => {
            let mut changed = false;
            for item in items {
                changed |= normalize_developer_roles(item);
            }
            changed
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => false,
    }
}

fn is_system_role(value: &Value) -> bool {
    value
        .as_object()
        .and_then(|object| object.get("role"))
        .and_then(Value::as_str)
        .is_some_and(|role| role == "system")
}

fn move_system_messages_to_front(value: &mut Value) -> bool {
    let Value::Array(items) = value else {
        return false;
    };

    let mut saw_non_system = false;
    let mut changed = false;
    for item in items.iter() {
        if is_system_role(item) {
            changed |= saw_non_system;
        } else {
            saw_non_system = true;
        }
    }

    if changed {
        items.sort_by_key(|item| !is_system_role(item));
    }
    changed
}

fn content_as_parts(content: Value) -> Vec<Value> {
    match content {
        Value::Array(parts) => parts,
        Value::String(text) => vec![serde_json::json!({
            "type": "input_text",
            "text": text
        })],
        Value::Null => Vec::new(),
        other => vec![other],
    }
}

fn append_message_content(target: &mut Value, source: &Value) {
    let Some(source_content) = source.as_object().and_then(|object| object.get("content")).cloned() else {
        return;
    };
    let mut source_parts = content_as_parts(source_content);
    if source_parts.is_empty() {
        return;
    }

    let Some(target_object) = target.as_object_mut() else {
        return;
    };
    let target_content = target_object
        .entry("content".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    if !target_content.is_array() {
        let current = std::mem::take(target_content);
        *target_content = Value::Array(content_as_parts(current));
    }
    if let Value::Array(target_parts) = target_content {
        target_parts.append(&mut source_parts);
    }
}

fn merge_system_messages(value: &mut Value) -> bool {
    let Value::Array(items) = value else {
        return false;
    };

    let Some(first_system_index) = items.iter().position(is_system_role) else {
        return false;
    };

    let mut merged = Vec::with_capacity(items.len());
    let mut first_system = items[first_system_index].clone();
    let mut changed = false;

    for (index, item) in items.iter().enumerate() {
        if index == first_system_index {
            continue;
        }
        if is_system_role(item) {
            append_message_content(&mut first_system, item);
            changed = true;
        } else {
            merged.push(item.clone());
        }
    }

    if changed {
        merged.insert(0, first_system);
        *items = merged;
    }
    changed
}

fn system_message(text: &str) -> Value {
    serde_json::json!({
        "type": "message",
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": text
            }
        ]
    })
}

fn prepend_system_message(input: &mut Value, instructions: &str) {
    match input {
        Value::Array(items) => items.insert(0, system_message(instructions)),
        Value::String(user_text) => {
            let user_text = std::mem::take(user_text);
            *input = Value::Array(vec![
                system_message(instructions),
                serde_json::json!({
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_text
                        }
                    ]
                }),
            ]);
        }
        _ => {}
    }
}

#[cfg(test)]
fn normalize_request_body(body: Bytes, config: &Config) -> Bytes {
    normalize_proxy_request_body(body, config).body
}

fn normalize_proxy_request_body(body: Bytes, config: &Config) -> NormalizedProxyRequest {
    let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
        return NormalizedProxyRequest {
            body,
            namespace: NamespaceNormalization::default(),
        };
    };

    let mut changed = false;
    let mut namespace = NamespaceNormalization::default();
    if let Some(object) = value.as_object_mut() {
        if let Some(model) = object.get("model").and_then(Value::as_str) {
            let resolved = resolve_model_alias(model, &config.model_aliases);
            if resolved != model {
                debug!(
                    model_before = %model,
                    model_after = %resolved,
                    "rewrote proxy request model alias"
                );
                object.insert("model".to_string(), Value::String(resolved));
                changed = true;
            }
        }
        let instructions = object.remove("instructions").and_then(|value| match value {
            Value::String(instructions) if !instructions.is_empty() => Some(instructions),
            other => {
                object.insert("instructions".to_string(), other);
                None
            }
        });
        if let Some(instructions) = instructions {
            if let Some(input) = object.get_mut("input") {
                prepend_system_message(input, &instructions);
                debug!("moved proxy request instructions into input");
                changed = true;
            } else {
                object.insert("instructions".to_string(), Value::String(instructions));
            }
        }
        if let Some(input) = object.get_mut("input") {
            changed |= normalize_developer_roles(input);
            changed |= move_system_messages_to_front(input);
            changed |= merge_system_messages(input);
        }
        changed |= flatten_namespace_tools_for_upstream(object, &mut namespace);
        changed |= rewrite_tool_choice_for_upstream(object, &namespace);
    }

    if !changed {
        return NormalizedProxyRequest { body, namespace };
    }
    NormalizedProxyRequest {
        body: serde_json::to_vec(&value).map_or(body, Bytes::from),
        namespace,
    }
}

fn record_alias_candidate(
    candidates: &mut HashMap<String, Option<NamespaceCallMapping>>,
    alias: String,
    mapping: NamespaceCallMapping,
) {
    candidates
        .entry(alias)
        .and_modify(|candidate| *candidate = None)
        .or_insert_with(|| Some(mapping));
}

fn register_unambiguous_aliases(
    normalization: &mut NamespaceNormalization,
    candidates: HashMap<String, Option<NamespaceCallMapping>>,
    top_level_names: &HashSet<String>,
) {
    for (alias, mapping) in candidates {
        if top_level_names.contains(&alias) {
            continue;
        }
        if let Some(mapping) = mapping {
            normalization.calls.entry(alias).or_insert(mapping);
        }
    }
}

fn raw_top_level_tool_names(tools: &[Value]) -> HashSet<String> {
    tools
        .iter()
        .filter_map(|tool| {
            let object = tool.as_object()?;
            if object.get("type").and_then(Value::as_str) == Some("namespace") {
                return None;
            }
            object.get("name").and_then(Value::as_str).map(str::to_string)
        })
        .collect()
}

fn raw_namespace_has_flat_name_collision(
    namespace_name: &str,
    function_members: &[&Value],
    top_level_names: &HashSet<String>,
) -> bool {
    function_members.iter().any(|member| {
        member.get("name").and_then(Value::as_str).is_some_and(|member_name| {
            top_level_names.contains(&model_visible_namespace_member_name(namespace_name, member_name))
        })
    })
}

fn record_single_member_namespace_container_candidate(
    container_candidates: &mut HashMap<String, Option<NamespaceCallMapping>>,
    namespace_name: &str,
    function_members: &[&Value],
) {
    if function_members.len() != 1 {
        return;
    }
    let Some(member_name) = function_members[0].get("name").and_then(Value::as_str) else {
        return;
    };
    let flat_name = model_visible_namespace_member_name(namespace_name, member_name);
    record_alias_candidate(
        container_candidates,
        namespace_name.to_string(),
        NamespaceCallMapping {
            member: NamespaceMemberName {
                namespace: namespace_name.to_string(),
                name: member_name.to_string(),
            },
            upstream_name: flat_name,
            strip_container_arguments: true,
        },
    );
}

fn flatten_namespace_tools_for_upstream(
    object: &mut serde_json::Map<String, Value>,
    normalization: &mut NamespaceNormalization,
) -> bool {
    let Some(tools) = object.get_mut("tools").and_then(Value::as_array_mut) else {
        return false;
    };

    let original_tools = tools.clone();
    let top_level_names = raw_top_level_tool_names(tools);
    let mut bare_member_candidates: HashMap<String, Option<NamespaceCallMapping>> = HashMap::new();
    let mut legacy_member_candidates: HashMap<String, Option<NamespaceCallMapping>> = HashMap::new();
    let mut alternate_member_candidates: HashMap<String, Option<NamespaceCallMapping>> = HashMap::new();
    let mut container_candidates: HashMap<String, Option<NamespaceCallMapping>> = HashMap::new();
    let mut upstream_tools = Vec::with_capacity(tools.len());
    let mut changed = false;

    for tool in std::mem::take(tools) {
        let Some(tool_object) = tool.as_object() else {
            upstream_tools.push(tool);
            continue;
        };
        if tool_object.get("type").and_then(Value::as_str) != Some("namespace") {
            upstream_tools.push(tool);
            continue;
        }
        let Some(namespace_name) = tool_object.get("name").and_then(Value::as_str) else {
            upstream_tools.push(tool);
            continue;
        };
        let Some(members) = tool_object.get("tools").and_then(Value::as_array) else {
            upstream_tools.push(tool);
            continue;
        };

        let function_members: Vec<&Value> = members
            .iter()
            .filter(|member| member.get("type").and_then(Value::as_str) == Some("function"))
            .collect();
        if function_members.is_empty() {
            upstream_tools.push(tool);
            continue;
        }
        if raw_namespace_has_flat_name_collision(namespace_name, &function_members, &top_level_names) {
            debug!(
                namespace = %namespace_name,
                "leaving raw namespace tool unflattened because a top-level tool uses a generated name"
            );
            upstream_tools.push(tool);
            continue;
        }

        let mut emitted_namespace_members = false;
        for member in &function_members {
            let Some(member_name) = member.get("name").and_then(Value::as_str) else {
                continue;
            };
            let flat_name = model_visible_namespace_member_name(namespace_name, member_name);
            debug!(
                namespace = %namespace_name,
                member = %member_name,
                upstream_name = %flat_name,
                "flattened raw namespace tool member for upstream"
            );
            let mut upstream_member = (*member).clone();
            if let Some(member_object) = upstream_member.as_object_mut() {
                member_object.insert("name".to_string(), Value::String(flat_name.clone()));
            }
            upstream_tools.push(upstream_member);
            let member = NamespaceMemberName {
                namespace: namespace_name.to_string(),
                name: member_name.to_string(),
            };
            let mapping = NamespaceCallMapping {
                member,
                upstream_name: flat_name.clone(),
                strip_container_arguments: false,
            };
            normalization.calls.insert(flat_name.clone(), mapping.clone());
            let legacy_name = legacy_model_visible_namespace_member_name(namespace_name, member_name);
            record_alias_candidate(&mut legacy_member_candidates, legacy_name, mapping.clone());
            let alternate_name = alternate_model_visible_namespace_member_name(namespace_name, member_name);
            record_alias_candidate(&mut alternate_member_candidates, alternate_name, mapping.clone());
            record_alias_candidate(&mut bare_member_candidates, member_name.to_string(), mapping);
            emitted_namespace_members = true;
            changed = true;
        }

        if !emitted_namespace_members {
            upstream_tools.push(tool);
            continue;
        }

        record_single_member_namespace_container_candidate(
            &mut container_candidates,
            namespace_name,
            &function_members,
        );
    }

    if changed {
        register_unambiguous_aliases(normalization, legacy_member_candidates, &top_level_names);
        register_unambiguous_aliases(normalization, alternate_member_candidates, &top_level_names);
        register_unambiguous_aliases(normalization, bare_member_candidates, &top_level_names);
        register_unambiguous_aliases(normalization, container_candidates, &top_level_names);
        normalization.original_tools = Some(Value::Array(original_tools));
        *tools = upstream_tools;
    } else {
        *tools = original_tools;
    }
    changed
}

fn strip_namespace_container_arguments(value: &mut Value) {
    let Some(arguments) = value.as_object_mut().and_then(|object| object.get_mut("arguments")) else {
        return;
    };
    let Some(arguments_text) = arguments.as_str() else {
        return;
    };
    let Ok(mut parsed) = serde_json::from_str::<Value>(arguments_text) else {
        return;
    };
    let Some(object) = parsed.as_object_mut() else {
        return;
    };
    if object.remove("tools").is_some() {
        *arguments = Value::String(serde_json::to_string(&parsed).unwrap_or_else(|_| arguments_text.to_string()));
    }
}

fn normalize_call_object(value: &mut Value, namespace: &NamespaceNormalization) -> bool {
    let Some(object) = value.as_object_mut() else {
        return false;
    };
    if object.get("type").and_then(Value::as_str) != Some("function_call") {
        return false;
    }
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return false;
    };
    if object.get("namespace").and_then(Value::as_str).is_some() {
        return false;
    }
    let Some(mapping) = namespace.calls.get(name) else {
        return false;
    };
    let original_name = name.to_string();

    object.insert("namespace".to_string(), Value::String(mapping.member.namespace.clone()));
    object.insert("name".to_string(), Value::String(mapping.member.name.clone()));
    if mapping.strip_container_arguments {
        strip_namespace_container_arguments(value);
    }
    debug!(
        upstream_name = %original_name,
        namespace = %mapping.member.namespace,
        member = %mapping.member.name,
        stripped_container_arguments = mapping.strip_container_arguments,
        "restored raw proxy namespace function call"
    );
    true
}

fn namespace_mapping_for_member<'a>(
    normalization: &'a NamespaceNormalization,
    namespace: &str,
    name: &str,
) -> Option<&'a NamespaceCallMapping> {
    normalization
        .calls
        .values()
        .find(|mapping| mapping.member.namespace == namespace && mapping.member.name == name)
}

fn rewrite_tool_choice_function_object(
    function: &mut serde_json::Map<String, Value>,
    normalization: &NamespaceNormalization,
) -> bool {
    let explicit_namespace = function.get("namespace").and_then(Value::as_str).map(str::to_string);
    let Some(name_text) = function.get("name").and_then(Value::as_str).map(str::to_string) else {
        return false;
    };
    let mapping = if let Some(namespace) = explicit_namespace.as_deref() {
        namespace_mapping_for_member(normalization, namespace, &name_text)
    } else {
        normalization.calls.get(&name_text)
    };
    let Some(mapping) = mapping else {
        return false;
    };

    let mut changed = mapping.upstream_name != name_text;
    if changed {
        function.insert("name".to_string(), Value::String(mapping.upstream_name.clone()));
    }
    if explicit_namespace.is_some() {
        changed |= function.remove("namespace").is_some();
    }
    changed
}

fn rewrite_tool_choice_for_upstream(
    object: &mut serde_json::Map<String, Value>,
    normalization: &NamespaceNormalization,
) -> bool {
    let Some(tool_choice) = object.get_mut("tool_choice") else {
        return false;
    };
    let Some(choice_object) = tool_choice.as_object_mut() else {
        return false;
    };

    if choice_object.get("type").and_then(Value::as_str) == Some("function") {
        return rewrite_tool_choice_function_object(choice_object, normalization);
    }

    choice_object
        .get_mut("function")
        .and_then(Value::as_object_mut)
        .is_some_and(|function| rewrite_tool_choice_function_object(function, normalization))
}

fn restore_original_tools(value: &mut Value, namespace: &NamespaceNormalization) -> bool {
    let Some(original_tools) = &namespace.original_tools else {
        return false;
    };
    let Some(object) = value.as_object_mut() else {
        return false;
    };
    if !object.get("tools").is_some_and(Value::is_array) {
        return false;
    }
    object.insert("tools".to_string(), original_tools.clone());
    true
}

fn normalize_response_value(value: &mut Value, namespace: &NamespaceNormalization) -> bool {
    let mut changed = false;
    changed |= restore_original_tools(value, namespace);

    if let Some(item) = value.as_object_mut().and_then(|object| object.get_mut("item")) {
        changed |= normalize_call_object(item, namespace);
    }

    changed |= normalize_call_object(value, namespace);

    for key in ["response", "payload"] {
        if let Some(nested) = value.as_object_mut().and_then(|object| object.get_mut(key)) {
            changed |= normalize_response_value(nested, namespace);
        }
    }

    if let Some(Value::Array(items)) = value.as_object_mut().and_then(|object| object.get_mut("output")) {
        for item in items {
            changed |= normalize_call_object(item, namespace);
        }
    }

    changed
}

fn normalize_sse_line(line: &str, namespace: &NamespaceNormalization) -> String {
    let Some(data) = line.strip_prefix("data: ") else {
        return line.to_string();
    };
    if data == "[DONE]" {
        return line.to_string();
    }
    let Ok(mut value) = serde_json::from_str::<Value>(data) else {
        return line.to_string();
    };
    if !normalize_response_value(&mut value, namespace) {
        return line.to_string();
    }
    serde_json::to_string(&value).map_or_else(|_| line.to_string(), |json| format!("data: {json}"))
}

fn normalize_response_body(body: Bytes, namespace: &NamespaceNormalization) -> Bytes {
    let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };
    if !normalize_response_value(&mut value, namespace) {
        return body;
    }
    serde_json::to_vec(&value).map_or(body, Bytes::from)
}

fn normalize_sse_stream(
    mut upstream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    namespace: NamespaceNormalization,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    Box::pin(stream! {
        let mut buffer = Vec::new();
        while let Some(chunk) = upstream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            buffer.extend_from_slice(&chunk);

            while let Some(pos) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = buffer.drain(..=pos).collect::<Vec<_>>();
                let line_end = if pos > 0 && line.get(pos - 1) == Some(&b'\r') {
                    pos - 1
                } else {
                    pos
                };
                let Ok(raw_line) = std::str::from_utf8(&line[..line_end]) else {
                    yield Ok(Bytes::from(line));
                    continue;
                };
                let normalized = normalize_sse_line(raw_line, &namespace);
                yield Ok(Bytes::from(format!("{normalized}\n")));
            }
        }

        if !buffer.is_empty() {
            let Ok(raw_line) = std::str::from_utf8(&buffer) else {
                yield Ok(Bytes::from(buffer));
                return;
            };
            let normalized = normalize_sse_line(raw_line, &namespace);
            yield Ok(Bytes::from(normalized));
        }
    })
}

#[must_use]
pub fn error_response(status: StatusCode, code: &str, message: &str) -> ProxyResponse {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "api_error",
            "param": null,
            "code": code,
        }
    });
    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    ProxyResponse {
        status,
        headers,
        body: ProxyBody::Full(Bytes::from(serde_json::to_vec(&body).unwrap_or_default())),
    }
}

/// Proxy a GET request to an arbitrary upstream path.
///
/// Applies the same header filtering and auth injection as [`proxy_request`].
/// Uses the non-streaming client; the response body is returned as a full
/// [`ProxyBody::Full`] payload.
pub async fn proxy_get(path: &str, request_headers: &HeaderMap, state: &ProxyState) -> ProxyResponse {
    let llm_headers = filter_request_headers(request_headers, &state.config);
    let base = state.config.llm_api_base.trim_end_matches('/');
    let url = format!("{base}/{}", path.trim_start_matches('/'));

    let llm_resp = match state.non_stream_client.get(&url).headers(llm_headers).send().await {
        Ok(r) => r,
        Err(e) if e.is_timeout() => {
            warn!("upstream GET {path} timed out: {e}");
            return error_response(StatusCode::GATEWAY_TIMEOUT, "upstream_timeout", "upstream timeout");
        }
        Err(e) => {
            warn!("upstream GET {path} failed: {e}");
            return error_response(StatusCode::BAD_GATEWAY, "upstream_unavailable", "upstream unavailable");
        }
    };

    let status = StatusCode::from_u16(llm_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let response_headers = filter_response_headers(llm_resp.headers());

    match llm_resp.bytes().await {
        Ok(payload) => ProxyResponse {
            status,
            headers: response_headers,
            body: ProxyBody::Full(payload),
        },
        Err(e) => {
            warn!("failed to read upstream GET {path} body: {e}");
            error_response(
                StatusCode::BAD_GATEWAY,
                "upstream_unavailable",
                "failed to read upstream response",
            )
        }
    }
}

pub async fn proxy_request(request: ProxyRequest, state: &ProxyState) -> ProxyResponse {
    let request_body_bytes = request.body.len();
    let is_streaming = serde_json::from_slice::<Value>(&request.body)
        .ok()
        .and_then(|v| v.get("stream")?.as_bool())
        .unwrap_or(false);

    let llm_headers = filter_request_headers(&request.headers, &state.config);
    let normalized_request = normalize_proxy_request_body(request.body, &state.config);
    let body = normalized_request.body;
    let namespace = normalized_request.namespace;

    let base = state.config.llm_api_base.trim_end_matches('/');
    let mut url = format!("{base}/v1/responses");
    if let Some(q) = &request.query {
        url.push('?');
        url.push_str(q);
    }
    debug!(
        url = %url,
        stream = is_streaming,
        request_body_bytes,
        normalized_body_bytes = body.len(),
        has_namespace_normalization = !namespace.is_empty(),
        "proxying responses request to upstream"
    );

    let client = if is_streaming {
        &state.stream_client
    } else {
        &state.non_stream_client
    };

    let llm_resp = match client.post(&url).headers(llm_headers).body(body).send().await {
        Ok(r) => r,
        Err(e) if e.is_timeout() => {
            warn!("LLM request timed out: {e}");
            return error_response(StatusCode::GATEWAY_TIMEOUT, "llm_timeout", "LLM timeout");
        }
        Err(e) => {
            warn!("LLM request failed: {e}");
            return error_response(StatusCode::BAD_GATEWAY, "llm_unavailable", "LLM unavailable");
        }
    };

    let status = StatusCode::from_u16(llm_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let response_is_sse = is_sse_content_type(llm_resp.headers());
    debug!(
        status = status.as_u16(),
        response_is_sse,
        has_namespace_normalization = !namespace.is_empty(),
        "received upstream proxy response"
    );
    let mut response_headers = filter_response_headers(llm_resp.headers());

    if response_is_sse {
        response_headers.insert("x-accel-buffering", HeaderValue::from_static("no"));

        let byte_stream: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> =
            Box::pin(llm_resp.bytes_stream().map_err(std::io::Error::other));
        let byte_stream = if namespace.is_empty() {
            byte_stream
        } else {
            debug!("normalizing namespace calls in upstream SSE proxy response");
            normalize_sse_stream(byte_stream, namespace)
        };

        return ProxyResponse {
            status,
            headers: response_headers,
            body: ProxyBody::Stream(Box::pin(byte_stream)),
        };
    }

    let mut payload: Bytes = match llm_resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            warn!("failed to read LLM response body: {e}");
            return error_response(
                StatusCode::BAD_GATEWAY,
                "llm_unavailable",
                "Failed to read LLM response",
            );
        }
    };
    if !namespace.is_empty() {
        debug!("normalizing namespace calls in upstream JSON proxy response");
        payload = normalize_response_body(payload, &namespace);
    }

    ProxyResponse {
        status,
        headers: response_headers,
        body: ProxyBody::Full(payload),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn test_config() -> Config {
        Config {
            llm_api_base: "http://localhost:8000".to_owned(),
            openai_api_key: Some("test-key".to_owned()),
            llm_ready_timeout_s: 5.0,
            llm_ready_interval_s: 0.1,
            skip_llm_ready_check: false,
            db_url: None,
            model_aliases: std::collections::HashMap::new(),
        }
    }

    fn test_config_no_key() -> Config {
        Config {
            openai_api_key: None,
            ..test_config()
        }
    }

    #[test]
    fn hop_by_hop_detected() {
        assert!(is_hop_by_hop("connection"));
        assert!(is_hop_by_hop("Connection"));
        assert!(is_hop_by_hop("keep-alive"));
        assert!(is_hop_by_hop("transfer-encoding"));
        assert!(is_hop_by_hop("proxy-authorization"));
    }

    #[test]
    fn non_hop_by_hop_passes() {
        assert!(!is_hop_by_hop("content-type"));
        assert!(!is_hop_by_hop("x-custom"));
        assert!(!is_hop_by_hop("authorization"));
    }

    #[test]
    fn request_drop_includes_host_and_content_length() {
        assert!(is_request_drop("host"));
        assert!(is_request_drop("content-length"));
        assert!(is_request_drop("connection"));
        assert!(!is_request_drop("content-type"));
    }

    #[test]
    fn filter_request_headers_strips_hop_by_hop() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("connection", "keep-alive".parse().unwrap());
        headers.insert("proxy-authorization", "Basic abc".parse().unwrap());
        headers.insert("x-custom", "value".parse().unwrap());

        let config = test_config_no_key();
        let filtered = filter_request_headers(&headers, &config);

        assert!(filtered.contains_key("content-type"));
        assert!(filtered.contains_key("x-custom"));
        assert!(!filtered.contains_key("connection"));
        assert!(!filtered.contains_key("proxy-authorization"));
    }

    #[test]
    fn filter_request_headers_strips_host_and_content_length() {
        let mut headers = HeaderMap::new();
        headers.insert("host", "example.com".parse().unwrap());
        headers.insert("content-length", "42".parse().unwrap());
        headers.insert("accept", "*/*".parse().unwrap());

        let config = test_config_no_key();
        let filtered = filter_request_headers(&headers, &config);

        assert!(!filtered.contains_key("host"));
        assert!(!filtered.contains_key("content-length"));
        assert!(filtered.contains_key("accept"));
    }

    #[test]
    fn auth_injected_when_no_client_auth() {
        let headers = HeaderMap::new();
        let config = test_config();
        let filtered = filter_request_headers(&headers, &config);

        assert_eq!(
            filtered.get("authorization").unwrap().to_str().unwrap(),
            "Bearer test-key"
        );
    }

    #[test]
    fn client_auth_takes_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer client-token".parse().unwrap());

        let config = test_config();
        let filtered = filter_request_headers(&headers, &config);

        assert_eq!(
            filtered.get("authorization").unwrap().to_str().unwrap(),
            "Bearer client-token"
        );
    }

    #[test]
    fn no_auth_injected_when_key_empty() {
        let headers = HeaderMap::new();
        let config = Config {
            openai_api_key: Some("  ".to_owned()),
            ..test_config()
        };
        let filtered = filter_request_headers(&headers, &config);

        assert!(!filtered.contains_key("authorization"));
    }

    #[test]
    fn no_auth_injected_when_key_none() {
        let headers = HeaderMap::new();
        let config = test_config_no_key();
        let filtered = filter_request_headers(&headers, &config);

        assert!(!filtered.contains_key("authorization"));
    }

    #[test]
    fn model_alias_rewrites_request_body() {
        let mut config = test_config();
        config
            .model_aliases
            .insert("codex-auto-review".to_string(), "real-model".to_string());
        let body = Bytes::from_static(br#"{"model":"codex-auto-review","input":"hi","store":false}"#);

        let rewritten = normalize_request_body(body, &config);
        let value: Value = serde_json::from_slice(&rewritten).unwrap();

        assert_eq!(value["model"], "real-model");
        assert_eq!(value["input"], "hi");
    }

    #[test]
    fn proxy_request_body_normalizes_developer_roles() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"model":"test","input":[{"role":"developer","content":"rules"},{"role":"user","content":"hi"}],"store":false}"#,
        );

        let rewritten = normalize_request_body(body, &config);
        let value: Value = serde_json::from_slice(&rewritten).unwrap();

        assert_eq!(value["input"][0]["role"], "system");
        assert_eq!(value["input"][1]["role"], "user");
    }

    #[test]
    fn proxy_request_body_moves_system_messages_to_front() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"model":"test","input":[{"role":"user","content":"hi"},{"role":"developer","content":"rules"}],"store":false}"#,
        );

        let rewritten = normalize_request_body(body, &config);
        let value: Value = serde_json::from_slice(&rewritten).unwrap();

        assert_eq!(value["input"][0]["role"], "system");
        assert_eq!(value["input"][1]["role"], "user");
    }

    #[test]
    fn proxy_request_body_moves_instructions_into_input() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"model":"test","instructions":"rules","input":[{"role":"user","content":"hi"}],"store":false}"#,
        );

        let rewritten = normalize_request_body(body, &config);
        let value: Value = serde_json::from_slice(&rewritten).unwrap();

        assert!(value.get("instructions").is_none());
        assert_eq!(value["input"][0]["role"], "system");
        assert_eq!(value["input"][0]["content"][0]["text"], "rules");
        assert_eq!(value["input"][1]["role"], "user");
    }

    #[test]
    fn proxy_request_body_moves_instructions_into_string_input() {
        let config = test_config();
        let body = Bytes::from_static(br#"{"model":"test","instructions":"rules","input":"hi","store":false}"#);

        let rewritten = normalize_request_body(body, &config);
        let value: Value = serde_json::from_slice(&rewritten).unwrap();

        assert!(value.get("instructions").is_none());
        assert_eq!(value["input"][0]["role"], "system");
        assert_eq!(value["input"][0]["content"][0]["text"], "rules");
        assert_eq!(value["input"][1]["role"], "user");
        assert_eq!(value["input"][1]["content"][0]["text"], "hi");
    }

    #[test]
    fn proxy_request_body_merges_system_messages() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"model":"test","input":[{"role":"system","content":[{"type":"input_text","text":"rules 1"}]},{"role":"system","content":[{"type":"input_text","text":"rules 2"}]},{"role":"user","content":"hi"}],"store":false}"#,
        );

        let rewritten = normalize_request_body(body, &config);
        let value: Value = serde_json::from_slice(&rewritten).unwrap();

        assert_eq!(value["input"].as_array().unwrap().len(), 2);
        assert_eq!(value["input"][0]["role"], "system");
        assert_eq!(value["input"][0]["content"][0]["text"], "rules 1");
        assert_eq!(value["input"][0]["content"][1]["text"], "rules 2");
        assert_eq!(value["input"][1]["role"], "user");
    }

    #[test]
    fn proxy_request_body_flattens_namespace_tools_for_upstream() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"model":"test","input":"hi","tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"echo_text","parameters":{"type":"object"}},{"type":"function","name":"add_numbers","parameters":{"type":"object"}}]}],"store":false}"#,
        );

        let normalized = normalize_proxy_request_body(body, &config);
        let value: Value = serde_json::from_slice(&normalized.body).unwrap();

        assert_eq!(value["tools"].as_array().unwrap().len(), 2);
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][0]["name"], "agentic_ns__mcp__agentic_fixture__echo_text");
        assert_eq!(value["tools"][1]["type"], "function");
        assert_eq!(
            value["tools"][1]["name"],
            "agentic_ns__mcp__agentic_fixture__add_numbers"
        );
        assert!(normalized.namespace.original_tools.is_some());
        assert!(
            normalized
                .namespace
                .calls
                .contains_key("agentic_ns__mcp__agentic_fixture__echo_text")
        );
    }

    #[test]
    fn proxy_request_body_rewrites_tool_choice_for_flattened_namespace_tool() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"model":"test","input":"hi","tools":[{"type":"namespace","name":"mcp__shell","tools":[{"type":"function","name":"run"}]}],"tool_choice":{"type":"function","name":"run"},"store":false}"#,
        );

        let normalized = normalize_proxy_request_body(body, &config);
        let value: Value = serde_json::from_slice(&normalized.body).unwrap();

        assert_eq!(value["tool_choice"]["type"], "function");
        assert_eq!(value["tool_choice"]["name"], "agentic_ns__mcp__shell__run");
    }

    #[test]
    fn proxy_request_body_rewrites_namespaced_tool_choice_for_flattened_namespace_tool() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"model":"test","input":"hi","tools":[{"type":"namespace","name":"mcp__shell","tools":[{"type":"function","name":"run"}]},{"type":"namespace","name":"mcp__git","tools":[{"type":"function","name":"run"}]}],"tool_choice":{"type":"function","namespace":"mcp__git","name":"run"},"store":false}"#,
        );

        let normalized = normalize_proxy_request_body(body, &config);
        let value: Value = serde_json::from_slice(&normalized.body).unwrap();

        assert_eq!(value["tool_choice"]["type"], "function");
        assert!(value["tool_choice"].get("namespace").is_none());
        assert_eq!(value["tool_choice"]["name"], "agentic_ns__mcp__git__run");
    }

    #[test]
    fn proxy_request_body_does_not_flatten_namespace_member_over_top_level_name() {
        let body = Bytes::from_static(
            br#"{"model":"test","input":"hi","tools":[{"type":"function","name":"agentic_ns__mcp__shell__run"},{"type":"namespace","name":"mcp__shell","tools":[{"type":"function","name":"run"}]},{"type":"namespace","name":"mcp__git","tools":[{"type":"function","name":"status"}]}],"tool_choice":{"type":"function","namespace":"mcp__shell","name":"run"},"store":false}"#,
        );

        let normalized = normalize_proxy_request_body(body, &test_config());
        let value: Value = serde_json::from_slice(&normalized.body).unwrap();
        let tools = value["tools"].as_array().unwrap();
        let flat_function_count = tools
            .iter()
            .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
            .filter(|tool| tool.get("name").and_then(Value::as_str) == Some("agentic_ns__mcp__shell__run"))
            .count();

        assert_eq!(flat_function_count, 1);
        assert!(tools.iter().any(|tool| {
            tool.get("type").and_then(Value::as_str) == Some("namespace")
                && tool.get("name").and_then(Value::as_str) == Some("mcp__shell")
        }));
        assert!(tools.iter().any(|tool| {
            tool.get("type").and_then(Value::as_str) == Some("function")
                && tool.get("name").and_then(Value::as_str) == Some("agentic_ns__mcp__git__status")
        }));
        assert_eq!(value["tool_choice"]["namespace"], "mcp__shell");
        assert_eq!(value["tool_choice"]["name"], "run");
        assert!(!normalized.namespace.calls.contains_key("agentic_ns__mcp__shell__run"));
    }

    #[test]
    fn proxy_request_body_keeps_colliding_namespace_whole_while_flattening_others() {
        for shell_choice in ["status", "run"] {
            let body = Bytes::from(
                serde_json::to_vec(&serde_json::json!({
                    "model": "test",
                    "input": "hi",
                    "tools": [
                        {"type": "function", "name": "agentic_ns__mcp__shell__run"},
                        {
                            "type": "namespace",
                            "name": "mcp__shell",
                            "tools": [
                                {"type": "function", "name": "run"},
                                {"type": "function", "name": "status"}
                            ]
                        },
                        {
                            "type": "namespace",
                            "name": "mcp__git",
                            "tools": [{"type": "function", "name": "status"}]
                        }
                    ],
                    "tool_choice": {
                        "type": "function",
                        "namespace": "mcp__shell",
                        "name": shell_choice
                    },
                    "store": false
                }))
                .unwrap(),
            );

            let normalized = normalize_proxy_request_body(body, &test_config());
            let value: Value = serde_json::from_slice(&normalized.body).unwrap();
            let tools = value["tools"].as_array().unwrap();
            let shell_namespace = tools
                .iter()
                .find(|tool| {
                    tool.get("type").and_then(Value::as_str) == Some("namespace")
                        && tool.get("name").and_then(Value::as_str) == Some("mcp__shell")
                })
                .expect("shell namespace");
            let shell_member_names: Vec<&str> = shell_namespace["tools"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|member| member.get("name").and_then(Value::as_str))
                .collect();

            assert_eq!(shell_member_names, vec!["run", "status"]);
            assert!(!tools.iter().any(|tool| {
                tool.get("type").and_then(Value::as_str) == Some("function")
                    && tool.get("name").and_then(Value::as_str) == Some("agentic_ns__mcp__shell__status")
            }));
            assert!(tools.iter().any(|tool| {
                tool.get("type").and_then(Value::as_str) == Some("function")
                    && tool.get("name").and_then(Value::as_str) == Some("agentic_ns__mcp__git__status")
            }));
            assert_eq!(value["tool_choice"]["namespace"], "mcp__shell");
            assert_eq!(value["tool_choice"]["name"], shell_choice);
            assert!(
                !normalized
                    .namespace
                    .calls
                    .values()
                    .any(|mapping| mapping.member.namespace == "mcp__shell")
            );
            assert!(normalized.namespace.calls.contains_key("agentic_ns__mcp__git__status"));
        }
    }

    #[test]
    fn proxy_sse_maps_flat_namespace_member_call() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"echo_text"},{"type":"function","name":"add_numbers"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &config);
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"agentic_ns__mcp__agentic_fixture__echo_text","call_id":"call_1","arguments":"{\"text\":\"hi\"}"}}"#;

        let normalized = normalize_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "echo_text");
        assert_eq!(value["item"]["arguments"], "{\"text\":\"hi\"}");
    }

    #[test]
    fn proxy_sse_does_not_namespace_non_function_call_item() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"custom_tool_call","name":"mcp__agentic_fixture.run","input":"patch"}}"#;

        let normalized = normalize_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert!(value["item"].get("namespace").is_none());
        assert_eq!(value["item"]["name"], "mcp__agentic_fixture.run");
        assert_eq!(value["item"]["type"], "custom_tool_call");
    }

    #[test]
    fn proxy_sse_flat_namespace_member_preserves_tools_argument() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &config);
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"agentic_ns__mcp__agentic_fixture__run","call_id":"call_1","arguments":"{\"tools\":\"legitimate\",\"cmd\":\"pwd\"}"}}"#;

        let normalized = normalize_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "run");
        assert_eq!(value["item"]["arguments"], "{\"tools\":\"legitimate\",\"cmd\":\"pwd\"}");
    }

    #[test]
    fn proxy_sse_maps_underscore_namespace_member_alias() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"echo_text"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"mcp__agentic_fixture_echo_text","call_id":"call_1","arguments":"{\"text\":\"hi\"}"}}"#;

        let normalized = normalize_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "echo_text");
    }

    #[test]
    fn proxy_sse_ambiguous_underscore_namespace_member_alias_is_not_normalized() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__a_b","tools":[{"type":"function","name":"c"}]},{"type":"namespace","name":"mcp__a","tools":[{"type":"function","name":"b_c"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"mcp__a_b_c","call_id":"call_1","arguments":"{}"}}"#;

        let normalized = normalize_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert!(value["item"].get("namespace").is_none());
        assert_eq!(value["item"]["name"], "mcp__a_b_c");
    }

    #[test]
    fn proxy_sse_maps_unambiguous_bare_namespace_member() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"run","call_id":"call_1","arguments":"{\"cmd\":\"pwd\"}"}}"#;

        let normalized = normalize_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "run");
    }

    #[test]
    fn proxy_json_restores_original_tools_and_maps_flat_call() {
        let config = test_config();
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"echo_text"},{"type":"function","name":"add_numbers"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &config);
        let upstream = Bytes::from_static(
            br#"{"id":"resp_1","object":"response","tools":[{"type":"function","name":"agentic_ns__mcp__agentic_fixture__echo_text"}],"output":[{"type":"function_call","name":"agentic_ns__mcp__agentic_fixture__echo_text","call_id":"call_1","arguments":"{\"text\":\"hi\"}"}]}"#,
        );

        let normalized = normalize_response_body(upstream, &normalized_request.namespace);
        let value: Value = serde_json::from_slice(&normalized).unwrap();

        assert_eq!(value["tools"][0]["type"], "namespace");
        assert_eq!(value["tools"][0]["name"], "mcp__agentic_fixture");
        assert_eq!(value["output"][0]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["output"][0]["name"], "echo_text");
    }

    #[test]
    fn proxy_sse_normalizes_namespace_container_call() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let line = r#"data: {"type":"response.output_item.done","item":{"type":"function_call","name":"mcp__agentic_fixture","call_id":"call_1","arguments":"{\"tools\":\"opaque\",\"cmd\":\"pwd\"}"}}"#;

        let normalized = normalize_sse_line(line, &normalized_request.namespace);
        let value: Value = serde_json::from_str(normalized.strip_prefix("data: ").unwrap()).unwrap();

        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "run");
        assert_eq!(value["item"]["arguments"], "{\"cmd\":\"pwd\"}");
    }

    #[tokio::test]
    async fn proxy_sse_stream_preserves_utf8_split_across_chunks() {
        let body = Bytes::from_static(
            br#"{"tools":[{"type":"namespace","name":"mcp__agentic_fixture","tools":[{"type":"function","name":"run"}]}]}"#,
        );
        let normalized_request = normalize_proxy_request_body(body, &test_config());
        let snowman = "\u{2603}";
        let line = format!(
            r#"data: {{"type":"response.output_item.done","item":{{"type":"function_call","name":"agentic_ns__mcp__agentic_fixture__run","call_id":"call_1","arguments":"{{\"text\":\"snow {snowman}\"}}"}}}}"#
        );
        let bytes = format!("{line}\n").into_bytes();
        let split_at = bytes
            .windows(snowman.len())
            .position(|window| window == snowman.as_bytes())
            .expect("snowman bytes present")
            + 1;
        let chunks = vec![
            Ok(Bytes::copy_from_slice(&bytes[..split_at])),
            Ok(Bytes::copy_from_slice(&bytes[split_at..])),
        ];
        let mut stream = normalize_sse_stream(Box::pin(futures::stream::iter(chunks)), normalized_request.namespace);

        let output = stream.next().await.expect("normalized line").expect("stream ok");
        assert!(stream.next().await.is_none());
        let text = String::from_utf8(output.to_vec()).expect("normalized line is utf8");
        assert!(!text.contains('\u{FFFD}'));

        let value: Value = serde_json::from_str(text.trim_end().strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(value["item"]["namespace"], "mcp__agentic_fixture");
        assert_eq!(value["item"]["name"], "run");
        assert_eq!(value["item"]["arguments"], format!(r#"{{"text":"snow {snowman}"}}"#));
    }

    #[test]
    fn filter_response_headers_strips_hop_by_hop() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("connection", "keep-alive".parse().unwrap());
        headers.insert("x-request-id", "abc".parse().unwrap());

        let filtered = filter_response_headers(&headers);

        assert!(filtered.contains_key("content-type"));
        assert!(filtered.contains_key("x-request-id"));
        assert!(!filtered.contains_key("connection"));
    }

    #[test]
    fn sse_content_type_detected() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("content-type", "text/event-stream; charset=utf-8".parse().unwrap());
        assert!(is_sse_content_type(&headers));
    }

    #[test]
    fn sse_content_type_case_insensitive() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("content-type", "Text/Event-Stream".parse().unwrap());
        assert!(is_sse_content_type(&headers));
    }

    #[test]
    fn non_sse_content_type_rejected() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        assert!(!is_sse_content_type(&headers));
    }

    #[test]
    fn missing_content_type_not_sse() {
        let headers = reqwest::header::HeaderMap::new();
        assert!(!is_sse_content_type(&headers));
    }
}
