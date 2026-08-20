use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::codex::insert_namespace_entries;
use super::custom::{CustomHandler, CustomToolMap, insert_custom_entry};
use super::executors::GatewayExecutors;
use super::function::insert_function_entry;
use super::mcp::handler::{McpToolMap, McpToolRef};
use super::mcp::registry::insert_discovered_mcp_entry;
use super::names::validate_model_visible_declared_names;
use super::search;
use super::web_search::insert_web_search_entry;
use super::{CodexNamespaceHandler, GatewayExecutor, McpHandler, NamespaceMap, ToolError, ToolOutput, ToolSearchState};
use crate::events::WireEvent;

use crate::types::io::OutputItem;
use crate::types::io::output::{FunctionToolCall, McpListTools};
use crate::types::tools::{CodeInterpreterToolParam, FileSearchToolParam, ResponsesTool};
use crate::utils::common::{serialize_to_value, serialize_to_value_or_custom_default};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
    ToolSearch,
    Custom,
    CodexNamespace,
    Mcp,
    /// Internal routing discriminant. Serializes as `"web_search"`.
    /// Note: the corresponding `ResponsesTool` wire tag is `"web_search_preview"`.
    /// `ToolType` is not used in wire-facing types so the names differ intentionally.
    WebSearch,
    FileSearch,
    CodeInterpreter,
}

impl ToolType {
    #[must_use]
    pub(crate) const fn description(self) -> &'static str {
        match self {
            Self::Function => "function tool",
            Self::ToolSearch => "tool search",
            Self::Custom => "custom tool",
            Self::CodexNamespace => "Codex namespace tool",
            Self::Mcp => "MCP tool",
            Self::WebSearch => "web search tool",
            Self::FileSearch => "file search tool",
            Self::CodeInterpreter => "code interpreter tool",
        }
    }

    #[must_use]
    pub const fn is_gateway_owned(self) -> bool {
        !matches!(
            self,
            Self::Function | Self::ToolSearch | Self::Custom | Self::CodexNamespace
        )
    }
}

/// Per-request routing entry keyed by the tool name the model will call.
#[derive(Clone)]
pub struct ToolEntry {
    pub tool_type: ToolType,
    /// Full serialised tool param for the executor (used during dispatch).
    pub config: Value,
    /// For MCP tools: which server this tool belongs to.
    pub server_label: Option<String>,
    pub handler: Option<Arc<dyn GatewayExecutor>>,
}

impl std::fmt::Debug for ToolEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolEntry")
            .field("tool_type", &self.tool_type)
            .field("config", &self.config)
            .field("server_label", &self.server_label)
            .field("handler", &self.handler.is_some())
            .finish()
    }
}

fn insert_unique_tool_entries(
    entries: &mut HashMap<String, ToolEntry>,
    insert: impl FnOnce(&mut HashMap<String, ToolEntry>),
) -> Result<(), ToolError> {
    let mut resolved = HashMap::new();
    insert(&mut resolved);
    for (name, entry) in resolved {
        match entries.entry(name) {
            Entry::Occupied(existing) => {
                return Err(ToolError::Config(format!(
                    "{} registry name '{}' conflicts with existing {}",
                    entry.tool_type.description(),
                    existing.key(),
                    existing.get().tool_type.description()
                )));
            }
            Entry::Vacant(vacant) => {
                vacant.insert(entry);
            }
        }
    }
    Ok(())
}

pub struct GatewayDispatchResult {
    pub tool_type: ToolType,
    pub output: Result<ToolOutput, ToolError>,
}

// TODO: move to a dedicated file_search module alongside its `ToolHandler`
// once file_search execution is implemented.
fn insert_file_search_entry(
    entries: &mut HashMap<String, ToolEntry>,
    p: &FileSearchToolParam,
    handler: Option<Arc<dyn GatewayExecutor>>,
) {
    serialize_to_value_or_custom_default(
        p,
        "file_search tool config serialization failed",
        |config| {
            entries.insert(
                "file_search".to_owned(),
                ToolEntry {
                    tool_type: ToolType::FileSearch,
                    config,
                    server_label: None,
                    handler,
                },
            );
        },
        (),
    );
}

// TODO: move to a dedicated code_interpreter module alongside its `ToolHandler`
// once code_interpreter execution is implemented.
fn insert_code_interpreter_entry(
    entries: &mut HashMap<String, ToolEntry>,
    p: &CodeInterpreterToolParam,
    handler: Option<Arc<dyn GatewayExecutor>>,
) {
    serialize_to_value_or_custom_default(
        p,
        "code_interpreter tool config serialization failed",
        |config| {
            entries.insert(
                "code_interpreter".to_owned(),
                ToolEntry {
                    tool_type: ToolType::CodeInterpreter,
                    config,
                    server_label: None,
                    handler,
                },
            );
        },
        (),
    );
}

/// Request-scoped registry built from `RequestPayload.tools`.
/// Maps the name the LLM sees → routing metadata.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    entries: HashMap<String, ToolEntry>,

    /// Request-scoped public response translation is active even when a
    /// declaration-free replay has no synthetic `tool_search` registry entry.
    tool_search_translation_enabled: bool,

    /// Exact model-visible function names known publicly but withheld from the
    /// effective private tool set.
    withheld_function_names: HashSet<String>,

    /// Built once from the declared tools, so final payload and streaming event
    /// restoration don't rebuild it on every call.
    namespace_map: Option<NamespaceMap>,

    /// Maps normalized custom function names back to their public declarations
    /// for response lifecycle metadata restoration.
    custom_tool_map: Option<CustomToolMap>,

    /// Maps model-visible MCP function names back to their public server and
    /// tool identities without reparsing executor configuration.
    mcp_tool_map: McpToolMap,

    /// Request-scoped MCP discovery output items retained in declaration order.
    mcp_list_tools_items: Vec<McpListTools>,
}

impl ToolRegistry {
    /// Build a registry from declared tools and attach gateway handlers for dispatchable tool types.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] when Codex namespace member flattening
    /// would collide with another declared tool name, or when discovered MCP
    /// tools derive the same internal model-visible name.
    ///
    /// # Panics
    ///
    /// Panics if serialization of a tool param struct fails, which cannot happen
    /// for the types defined in this module (`#[derive(Serialize)]` on plain structs).
    pub async fn build_with_handlers(
        tools: &mut [ResponsesTool],
        executors: &mut GatewayExecutors,
    ) -> Result<Self, ToolError> {
        let mut entries = HashMap::with_capacity(tools.len());
        let mut mcp_tool_map = McpToolMap::default();
        let mut mcp_list_tools_items = Vec::new();
        // Validate declaration-derived names before MCP I/O, then key namespace
        // members by the same flat name used in the private inference request.
        // Discovered MCP names retain their post-list collision pass.
        validate_model_visible_declared_names(tools)?;
        let resolved_tools = CodexNamespaceHandler::resolve_namespace_members_after_validation(tools);
        McpHandler::validate_server_labels(&resolved_tools)?;

        for (index, tool) in resolved_tools.iter().enumerate() {
            match tool {
                ResponsesTool::Function(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| insert_function_entry(resolved, p))?;
                }
                ResponsesTool::ToolSearch(_) => {
                    tracing::debug!("client-executed tool_search declaration is omitted from the dispatch registry");
                }
                ResponsesTool::Mcp(p) => {
                    let tool_set = match executors.mcp_server_tools(p).await {
                        Ok(tool_set) => tool_set,
                        Err(error) => {
                            mcp_list_tools_items.push(McpHandler::failed_list_tools_item(&p.server_label, &error));
                            continue;
                        }
                    };
                    let handlers = tool_set.discovered_handlers;
                    mcp_list_tools_items.push(tool_set.list_tools_item);
                    if let ResponsesTool::Mcp(declaration) = &mut tools[index] {
                        declaration.discovered_tools = handlers.iter().map(|item| item.param.clone()).collect();
                    }
                    for discovered in handlers {
                        let internal_name = discovered.param.internal_name.clone();
                        let tool_ref = McpToolRef::from(&discovered.param);
                        insert_unique_tool_entries(&mut entries, |resolved| {
                            insert_discovered_mcp_entry(resolved, discovered);
                        })?;
                        mcp_tool_map.record(internal_name, tool_ref);
                    }
                }
                ResponsesTool::WebSearch(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| {
                        insert_web_search_entry(resolved, p, executors.web_search_handler());
                    })?;
                }
                ResponsesTool::FileSearch(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| insert_file_search_entry(resolved, p, None))?;
                }
                ResponsesTool::CodeInterpreter(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| {
                        insert_code_interpreter_entry(resolved, p, None);
                    })?;
                }
                ResponsesTool::Namespace(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| insert_namespace_entries(resolved, p))?;
                }
                ResponsesTool::Custom(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| insert_custom_entry(resolved, p))?;
                }
                ResponsesTool::Unknown => {
                    tracing::debug!("unknown tool declared but skipped in registry");
                }
            }
        }

        let namespace_map = CodexNamespaceHandler.build_namespace_map((!tools.is_empty()).then_some(tools))?;
        let custom_tool_map = CustomHandler::build_tool_map(tools);

        Ok(Self {
            entries,
            tool_search_translation_enabled: false,
            withheld_function_names: HashSet::new(),
            namespace_map,
            custom_tool_map,
            mcp_tool_map,
            mcp_list_tools_items,
        })
    }

    #[must_use]
    pub fn lookup(&self, tool_name: &str) -> Option<&ToolEntry> {
        self.entries.get(tool_name)
    }

    pub(crate) fn tool_type_map(&self) -> HashMap<String, ToolType> {
        let mut tool_types = self
            .entries
            .iter()
            .map(|(name, entry)| (name.clone(), entry.tool_type))
            .collect::<HashMap<_, _>>();
        if self.tool_search_translation_enabled {
            tool_types.insert("tool_search".to_owned(), ToolType::ToolSearch);
        }
        tool_types
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn contains_mcp_server_label(&self, server_label: &str) -> bool {
        self.mcp_tool_map.contains_server_label(server_label)
    }

    pub(crate) fn mcp_tool_ref(&self, internal_name: &str) -> Option<&McpToolRef> {
        self.mcp_tool_map.tool_ref(internal_name)
    }

    #[must_use]
    pub(crate) fn mcp_list_tools_items(&self) -> &[McpListTools] {
        &self.mcp_list_tools_items
    }

    /// Reclassify only the synthetic function prepared by active request-scoped
    /// tool-search state. An inactive ordinary function named `tool_search`
    /// remains an ordinary client function.
    pub(crate) fn classify_tool_search(&mut self, state: &ToolSearchState) -> Result<(), ToolError> {
        self.tool_search_translation_enabled = state.is_active();
        self.withheld_function_names.clone_from(state.withheld_function_names());
        if !self.tool_search_translation_enabled {
            return Ok(());
        }
        if self
            .entries
            .keys()
            .any(|name| self.withheld_function_names.contains(name))
        {
            return Err(ToolError::Config(
                "a loaded tool collides with a withheld function name".to_owned(),
            ));
        }
        if self
            .mcp_tool_map
            .resolves_call_before_load(state.mcp_load_positions(), state.unqualified_call_positions())
        {
            return Err(ToolError::Config(
                "request history calls an MCP function before its server definition is loaded".to_owned(),
            ));
        }
        let Some(synthetic) = state.synthetic_function() else {
            return Ok(());
        };
        let entry = self.entries.get_mut(&synthetic.name).ok_or_else(|| {
            ToolError::Config("prepared tool-search function is missing from the private registry".to_owned())
        })?;
        if entry.tool_type != ToolType::Function || entry.handler.is_some() {
            return Err(ToolError::Config(
                "prepared tool-search function has invalid private registry ownership".to_owned(),
            ));
        }
        entry.tool_type = ToolType::ToolSearch;
        Ok(())
    }

    #[must_use]
    fn has_tool_search(&self) -> bool {
        self.tool_search_translation_enabled
    }

    #[must_use]
    pub(crate) fn withheld_function_names(&self) -> &HashSet<String> {
        &self.withheld_function_names
    }

    /// Strictly convert a normalized blocking tool-search call to its public
    /// representation before permissive response rehydration can discard or
    /// default malformed fields.
    pub(crate) fn normalize_blocking_response(&self, response: &mut Value) -> Result<(), ToolError> {
        if !self.has_tool_search() {
            return Ok(());
        }
        let Some(items) = response.get_mut("output").and_then(Value::as_array_mut) else {
            return Ok(());
        };
        let mut search_calls = 0_u8;
        let mut public_ids = HashSet::with_capacity(items.len());
        let mut replacements = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let Some(object) = item.as_object() else {
                continue;
            };
            if object
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| self.withheld_function_names.contains(name))
            {
                return Err(search::invalid_upstream_withheld_function_call());
            }
            let reserved = object.get("name").and_then(Value::as_str) == Some("tool_search");
            let public_id = if reserved {
                search_calls = search_calls.saturating_add(1);
                if search_calls > 1 {
                    return Err(search::invalid_upstream_search_call());
                }
                let replacement = normalize_raw_search_call(object)?;
                let public_id = replacement["id"].as_str().unwrap_or_default().to_owned();
                replacements.push((index, replacement));
                public_id
            } else {
                object
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.trim().is_empty())
                    .unwrap_or_default()
                    .to_owned()
            };
            if !public_id.is_empty() && !public_ids.insert(public_id) {
                return Err(search::invalid_upstream_search_call());
            }
        }
        for (index, replacement) in replacements {
            items[index] = replacement;
        }
        Ok(())
    }

    /// Restore provider-normalized output types that are not already converted
    /// by the strict raw blocking-response seam.
    ///
    /// # Errors
    ///
    /// Reserved for restoration failures reported by output adapters.
    pub fn restore_final_payload_output(&self, output: &mut [OutputItem]) -> Result<(), ToolError> {
        if self.has_tool_search() {
            let mut search_calls = 0_u8;
            let mut public_ids = HashSet::with_capacity(output.len());
            let mut replacements = Vec::new();
            for (index, item) in output.iter().enumerate() {
                if matches!(item, OutputItem::FunctionCall(call) if self.withheld_function_names.contains(&call.name)) {
                    return Err(search::invalid_upstream_withheld_function_call());
                }
                let replacement = match item {
                    OutputItem::FunctionCall(call) if call.name == "tool_search" => {
                        search_calls = search_calls.saturating_add(1);
                        if search_calls > 1
                            || call.status != crate::types::event::MessageStatus::Completed
                            || call.namespace.is_some()
                        {
                            return Err(search::invalid_upstream_search_call());
                        }
                        Some(search::public_output_item(&call.id, &call.call_id, &call.arguments)?)
                    }
                    _ => None,
                };
                let candidate = replacement.as_ref().unwrap_or(item);
                let public_id =
                    serialize_to_value(candidate).map_err(|_| search::invalid_upstream_search_call())?["id"]
                        .as_str()
                        .filter(|id| !id.trim().is_empty())
                        .unwrap_or_default()
                        .to_owned();
                if !public_id.is_empty() && !public_ids.insert(public_id) {
                    return Err(search::invalid_upstream_search_call());
                }
                if let Some(replacement) = replacement {
                    replacements.push((index, replacement));
                }
            }
            for (index, replacement) in replacements {
                output[index] = replacement;
            }
        }
        CodexNamespaceHandler.restore_output_items(output, self.namespace_map.as_ref());
        Ok(())
    }

    pub fn restore_stream_event_wire(&self, wire: &mut WireEvent) -> bool {
        let custom_restored = CustomHandler::restore_response_wire(wire, self.custom_tool_map.as_ref());
        CodexNamespaceHandler.restore_response_wire(wire, self.namespace_map.as_ref()) | custom_restored
    }

    /// Returns the subset of `calls` whose names map to gateway-owned tools.
    #[must_use]
    pub fn gateway_owned<'a>(&self, calls: &'a [FunctionToolCall]) -> Vec<&'a FunctionToolCall> {
        calls
            .iter()
            .filter(|c| {
                self.entries
                    .get(&c.name)
                    .is_some_and(|e| e.tool_type.is_gateway_owned())
            })
            .collect()
    }

    #[must_use]
    pub fn is_gateway_owned_name(&self, name: &str) -> bool {
        self.entries
            .get(name)
            .is_some_and(|entry| entry.tool_type.is_gateway_owned())
    }

    /// Returns the subset of `calls` whose names map to client-owned tools
    /// (`Function`, Codex namespace members, or unknown names).
    #[must_use]
    pub fn client_owned<'a>(&self, calls: &'a [FunctionToolCall]) -> Vec<&'a FunctionToolCall> {
        calls
            .iter()
            .filter(|c| {
                self.entries
                    .get(&c.name)
                    .is_none_or(|e| !e.tool_type.is_gateway_owned())
            })
            .collect()
    }

    pub async fn dispatch(&self, call: &FunctionToolCall) -> Option<GatewayDispatchResult> {
        let entry = self.entries.get(&call.name)?;
        let handler = entry.handler.clone()?;
        let tool_type = entry.tool_type;
        let config = entry.config.clone();
        Some(GatewayDispatchResult {
            tool_type,
            output: handler
                .execute(&call.call_id, &call.name, &call.arguments, &config)
                .await,
        })
    }
}

fn normalize_raw_search_call(object: &serde_json::Map<String, Value>) -> Result<Value, ToolError> {
    let public = search::public_output_item_from_raw(object)?;
    serialize_to_value(&public).map_err(|_| search::invalid_upstream_search_call())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::executors::GatewayExecutorRegistration;
    use crate::tool::mcp::{McpDiscoveredHandler, McpHandler};
    use crate::types::event::MessageStatus;
    use crate::types::tools::McpDiscoveredToolParam;

    fn declaration(server_label: &str) -> ResponsesTool {
        serde_json::from_value(serde_json::json!({
            "type": "mcp",
            "server_label": server_label,
            "server_url": "http://127.0.0.1:8000/mcp",
            "require_approval": "never"
        }))
        .expect("MCP declaration")
    }

    fn discovered_handler(server_label: &str, tool_name: &str, internal_name: &str) -> McpDiscoveredHandler {
        let param = McpDiscoveredToolParam {
            server_label: server_label.to_owned(),
            tool_name: tool_name.to_owned(),
            internal_name: internal_name.to_owned(),
            tool: serde_json::from_value(serde_json::json!({
                "name": tool_name,
                "description": "Discovered test tool",
                "inputSchema": {"type": "object"}
            }))
            .expect("discovered MCP tool"),
        };
        McpDiscoveredHandler {
            param,
            handler: Arc::new(McpHandler::discovered_tool_spec_only()),
        }
    }

    #[tokio::test]
    async fn tool_search_declaration_has_no_registry_entry_or_handler() {
        let mut tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([{
            "type": "tool_search",
            "execution": "client",
            "description": "Find a tool",
            "parameters": {"type": "object"}
        }]))
        .expect("tool-search declaration");
        let mut executors = GatewayExecutors::default();

        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("inert declaration does not require a handler");

        assert!(registry.lookup("tool_search").is_none());
        assert!(registry.entries.is_empty());
    }

    #[tokio::test]
    async fn synthetic_tool_search_is_client_owned_never_dispatched_and_normalizes_blocking_output() {
        let (request, state) = prepared_search_state();
        let mut tools = private_tools(&state, &request);
        let mut executors = GatewayExecutors::default();
        let mut registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("synthetic function builds normally");

        registry
            .classify_tool_search(&state)
            .expect("prepared synthetic entry is reclassified exactly once");
        let entry = registry.lookup("tool_search").expect("classified search entry");
        assert_eq!(entry.tool_type, ToolType::ToolSearch);
        assert!(!entry.tool_type.is_gateway_owned());

        let call: FunctionToolCall = serde_json::from_value(serde_json::json!({
            "type": "function_call",
            "id": "fc_search_1",
            "call_id": "call_search_1",
            "name": "tool_search",
            "arguments": "{\"query\":\"weather\"}",
            "status": "completed"
        }))
        .expect("normalized call");
        assert!(
            registry.dispatch(&call).await.is_none(),
            "tool search has no gateway handler"
        );

        let mut response = serde_json::json!({"output": [{
            "type": "function_call",
            "id": "fc_search_1",
            "call_id": "call_search_1",
            "name": "tool_search",
            "namespace": null,
            "arguments": "{\"query\":\"weather\"}",
            "status": "completed"
        }]});
        registry
            .normalize_blocking_response(&mut response)
            .expect("valid normalized search call converts once");
        assert_eq!(
            response["output"][0],
            serde_json::json!({
                "type": "tool_search_call",
                "id": "tsc_search_1",
                "call_id": "call_search_1",
                "execution": "client",
                "arguments": {"query": "weather"},
                "status": "completed"
            })
        );
    }

    #[tokio::test]
    async fn raw_blocking_tool_search_normalization_is_strict_atomic_and_state_driven() {
        let (request, state) = prepared_search_state();
        let mut tools = private_tools(&state, &request);
        let mut registry = ToolRegistry::build_with_handlers(&mut tools, &mut GatewayExecutors::default())
            .await
            .expect("registry");
        registry.classify_tool_search(&state).expect("classification");

        let mut valid = serde_json::json!({"output": [{
            "type": "function_call",
            "id": "fc_search",
            "call_id": "call_search",
            "name": "tool_search",
            "namespace": null,
            "arguments": "{\"query\":\"weather\"}",
            "status": "completed"
        }]});
        registry
            .normalize_blocking_response(&mut valid)
            .expect("valid raw call normalizes once");
        assert_eq!(
            valid["output"][0],
            serde_json::json!({
                "type": "tool_search_call",
                "id": "tsc_search",
                "call_id": "call_search",
                "execution": "client",
                "arguments": {"query": "weather"},
                "status": "completed"
            })
        );

        let invalid_items = [
            serde_json::json!({"type":"custom_tool_call","id":"fc_search","call_id":"call_search","name":"tool_search","arguments":"{}"}),
            serde_json::json!({"type":"function_call","call_id":"call_search","name":"tool_search","arguments":"{}","status":"completed"}),
            serde_json::json!({"type":"function_call","id":" ","call_id":"call_search","name":"tool_search","arguments":"{}","status":"completed"}),
            serde_json::json!({"type":"function_call","id":"fc_search","name":"tool_search","arguments":"{}","status":"completed"}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":"","name":"tool_search","arguments":"{}","status":"completed"}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":7,"name":"tool_search","arguments":"{}","status":"completed"}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":"call_search","name":"tool_search","status":"completed"}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":"call_search","name":"tool_search","arguments":null,"status":"completed"}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":"call_search","name":"tool_search","arguments":"{","status":"completed"}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":"call_search","name":"tool_search","arguments":"[]","status":"completed"}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":"call_search","name":"tool_search","arguments":"null","status":"completed"}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":"call_search","name":"tool_search","arguments":"{}"}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":"call_search","name":"tool_search","arguments":"{}","status":"in_progress"}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":"call_search","name":"tool_search","arguments":"{}","status":7}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":"call_search","name":"tool_search","namespace":"tools","arguments":"{}","status":"completed"}),
            serde_json::json!({"type":"function_call","id":"fc_search","call_id":"call_search","name":"tool_search","namespace":7,"arguments":"{}","status":"completed"}),
        ];
        for item in invalid_items {
            let mut response = serde_json::json!({"output": [item]});
            assert!(
                matches!(
                    registry.normalize_blocking_response(&mut response),
                    Err(ToolError::Execution(message)) if message == "upstream returned an invalid tool-search call"
                ),
                "malformed reserved raw call must fail atomically"
            );
        }

        let raw_valid = serde_json::json!({
            "type":"function_call", "id":"fc_search", "call_id":"call_search", "name":"tool_search",
            "namespace":null, "arguments":"{}", "status":"completed"
        });
        let mut duplicate = serde_json::json!({"output": [raw_valid.clone(), raw_valid.clone()]});
        assert!(registry.normalize_blocking_response(&mut duplicate).is_err());
        let mut collision = serde_json::json!({"output": [
            raw_valid,
            {"type":"message","id":"tsc_search","role":"assistant","content":[]}
        ]});
        assert!(registry.normalize_blocking_response(&mut collision).is_err());

        let mut ordinary_tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([{
            "type": "function", "name": "tool_search", "parameters": {"type": "object"}
        }]))
        .expect("ordinary reserved-name function is valid while search is inactive");
        let ordinary = ToolRegistry::build_with_handlers(&mut ordinary_tools, &mut GatewayExecutors::default())
            .await
            .expect("ordinary registry");
        let mut inactive_response = serde_json::json!({"output": [invalid_items_for_inactive()]});
        ordinary
            .normalize_blocking_response(&mut inactive_response)
            .expect("inactive ordinary function is never name-only validated as search");
        assert_eq!(inactive_response["output"][0]["type"], "function_call");
    }

    #[tokio::test]
    async fn declaration_free_replay_enables_state_driven_raw_normalization() {
        let (request, state) = replayed_search_state();
        assert!(state.is_active());
        assert!(state.synthetic_function().is_none());
        let mut tools = private_tools(&state, &request);
        let mut registry = ToolRegistry::build_with_handlers(&mut tools, &mut GatewayExecutors::default())
            .await
            .expect("loaded function registry");
        assert!(registry.lookup("tool_search").is_none());
        registry
            .classify_tool_search(&state)
            .expect("enable replay translation");

        let mut valid = serde_json::json!({"output": [{
            "type": "function_call", "id": "fc_search", "call_id": "call_search",
            "name": "tool_search", "namespace": null, "arguments": "{\"query\":\"news\"}",
            "status": "completed"
        }]});
        registry
            .normalize_blocking_response(&mut valid)
            .expect("active replay translates reserved call");
        assert_eq!(valid["output"][0]["type"], "tool_search_call");

        let mut malformed = serde_json::json!({"output": [{
            "type": "function_call", "id": "fc_search", "call_id": "call_search",
            "name": "tool_search", "namespace": null, "arguments": "{}", "status": "in_progress"
        }]});
        assert!(matches!(
            registry.normalize_blocking_response(&mut malformed),
            Err(ToolError::Execution(_))
        ));
    }

    #[tokio::test]
    async fn withheld_namespace_guard_is_exact_not_prefix_based() {
        let request = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "find a tool",
            "tools": [
                {
                    "type": "tool_search", "execution": "client", "description": "Find a tool",
                    "parameters": {"type": "object"}
                },
                {
                    "type": "namespace", "name": "weather", "tools": [{
                        "type": "function", "name": "forecast", "defer_loading": true
                    }]
                },
                {
                    "type": "function", "name": "agentic_ns__weather__ordinary_prefix_like",
                    "parameters": {"type": "object"}
                }
            ],
            "store": false,
            "stream": false
        }))
        .expect("active namespace request");
        let state = ToolSearchState::build(&request).expect("prepared namespace state");
        let mut tools = private_tools(&state, &request);
        let mut registry = ToolRegistry::build_with_handlers(&mut tools, &mut GatewayExecutors::default())
            .await
            .expect("private registry");
        registry.classify_tool_search(&state).expect("classification");

        let mut ordinary = serde_json::json!({"output": [{
            "type": "function_call", "id": "fc_ordinary", "call_id": "call_ordinary",
            "name": "agentic_ns__weather__ordinary_prefix_like", "arguments": "{}", "status": "completed"
        }]});
        registry
            .normalize_blocking_response(&mut ordinary)
            .expect("unrelated prefix-like function remains ordinary");
        assert_eq!(ordinary["output"][0]["type"], "function_call");

        let mut withheld = serde_json::json!({"output": [{
            "type": "function_call", "id": "fc_withheld", "call_id": "call_withheld",
            "name": "agentic_ns__weather__forecast", "arguments": "{}", "status": "completed"
        }]});
        assert!(matches!(
            registry.normalize_blocking_response(&mut withheld),
            Err(ToolError::Execution(_))
        ));
    }

    fn invalid_items_for_inactive() -> Value {
        serde_json::json!({
            "type": "function_call",
            "id": "fc_ordinary",
            "call_id": "",
            "name": "tool_search",
            "arguments": "[]"
        })
    }

    fn private_tools(
        state: &ToolSearchState,
        request: &crate::types::request_response::RequestPayload,
    ) -> Vec<ResponsesTool> {
        state
            .private_inference_request(request)
            .expect("prepared state materializes a private inference request")
            .tools
            .expect("private tools")
    }

    fn prepared_search_state() -> (crate::types::request_response::RequestPayload, ToolSearchState) {
        let request: crate::types::request_response::RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "find a tool",
            "tools": [{
                "type": "tool_search",
                "execution": "client",
                "description": "Search the client catalog",
                "parameters": {"type": "object"}
            }],
            "store": false,
            "stream": false
        }))
        .expect("public request");
        let state = ToolSearchState::build(&request).expect("prepared search state");
        (request, state)
    }

    fn replayed_search_state() -> (crate::types::request_response::RequestPayload, ToolSearchState) {
        let request = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": [
                {
                    "type": "tool_search_call", "id": "tsc_1", "call_id": "call_1",
                    "execution": "client", "arguments": {"query": "weather"}, "status": "completed"
                },
                {
                    "type": "tool_search_output", "call_id": "call_1", "execution": "client",
                    "status": "completed", "tools": [{
                        "type": "function", "name": "get_weather", "parameters": {"type": "object"},
                        "defer_loading": true
                    }]
                }
            ],
            "tools": [],
            "store": false,
            "stream": false
        }))
        .expect("declaration-free replay request");
        let state = ToolSearchState::build(&request).expect("prepared replay state");
        (request, state)
    }

    #[test]
    fn declaration_free_replay_keeps_stream_translation_active_without_a_dispatch_entry() {
        let (_, state) = replayed_search_state();
        let mut registry = ToolRegistry::default();

        registry.classify_tool_search(&state).expect("replay classification");

        assert!(registry.lookup("tool_search").is_none());
        assert_eq!(registry.tool_type_map().get("tool_search"), Some(&ToolType::ToolSearch));
    }

    #[test]
    fn streaming_search_id_collision_is_atomic() {
        let (_, state) = replayed_search_state();
        let mut registry = ToolRegistry::default();
        registry.classify_tool_search(&state).expect("replay classification");
        let mut output: Vec<OutputItem> = serde_json::from_value(serde_json::json!([
            {
                "type": "function_call", "id": "fc_same", "call_id": "call_search",
                "name": "tool_search", "arguments": "{}", "status": "completed"
            },
            {
                "type": "tool_search_call", "id": "tsc_same", "call_id": "call_existing",
                "execution": "client", "arguments": {}, "status": "completed"
            }
        ]))
        .expect("output items");
        let before = serialize_to_value(&output).expect("before serializes");

        let error = registry
            .restore_final_payload_output(&mut output)
            .expect_err("public ID collision must fail");

        assert!(matches!(
            error,
            ToolError::Execution(message) if message == "upstream returned an invalid tool-search call"
        ));
        assert_eq!(serialize_to_value(&output).expect("after serializes"), before);
    }

    fn mixed_tool_declarations() -> Vec<ResponsesTool> {
        serde_json::from_value(serde_json::json!([
            {
                "type": "function",
                "name": "echo",
                "parameters": {"type": "object"}
            },
            {
                "type": "mcp",
                "server_label": "counter",
                "server_url": "http://127.0.0.1:8000/mcp",
                "require_approval": "never"
            },
            {"type": "web_search_preview", "search_context_size": "low"},
            {"type": "file_search", "vector_store_ids": ["vs_test"]},
            {"type": "code_interpreter"},
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run"}]
            },
            {"type": "custom", "name": "freeform"},
            {"type": "future_tool", "opaque": true}
        ]))
        .expect("mixed tool declarations")
    }

    fn assert_namespace_call_restoration(registry: &ToolRegistry) {
        let mut output = vec![OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "agentic_ns__mcp__shell__run".to_owned(),
            namespace: None,
            arguments: "{}".to_owned(),
            status: MessageStatus::Completed,
        })];
        registry
            .restore_final_payload_output(&mut output)
            .expect("ordinary namespace restoration succeeds");
        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected restored function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__shell"));
        assert_eq!(call.name, "run");
    }

    fn assert_mcp_list_tools_metadata(registry: &ToolRegistry) {
        let [list_tools] = registry.mcp_list_tools_items() else {
            panic!("expected one MCP list-tools item");
        };
        assert!(list_tools.id.starts_with("mcpl_"));
        assert_eq!(list_tools.server_label, "counter");
        assert_eq!(
            list_tools
                .tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["increment", "get_value"]
        );
        assert_eq!(list_tools.tools[0].description.as_deref(), Some("Discovered test tool"));
        assert_eq!(list_tools.tools[0].input_schema, serde_json::json!({"type": "object"}));
        assert_eq!(
            list_tools.tools[0].annotations,
            Some(serde_json::json!({"read_only": false}))
        );
    }

    #[tokio::test]
    async fn build_with_handlers_registers_mixed_tools_and_runtime_metadata() {
        let mut executors = GatewayExecutors::from_env(Arc::new(reqwest::Client::new()));
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "counter".to_owned(),
            handlers: vec![
                discovered_handler("counter", "increment", "mcp__counter__increment"),
                discovered_handler("counter", "get_value", "mcp__counter__get_value"),
            ],
        });
        let mut tools = mixed_tool_declarations();

        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("mixed registry");

        assert_eq!(registry.len(), 8);
        assert!(registry.contains_mcp_server_label("counter"));
        assert!(!registry.contains_mcp_server_label("missing"));
        assert_mcp_list_tools_metadata(&registry);

        let expected_entries = [
            ("echo", ToolType::Function, None, false),
            ("freeform", ToolType::Custom, None, false),
            ("mcp__counter__increment", ToolType::Mcp, Some("counter"), true),
            ("mcp__counter__get_value", ToolType::Mcp, Some("counter"), true),
            ("web_search", ToolType::WebSearch, None, true),
            ("file_search", ToolType::FileSearch, None, false),
            ("code_interpreter", ToolType::CodeInterpreter, None, false),
            (
                "agentic_ns__mcp__shell__run",
                ToolType::CodexNamespace,
                Some("mcp__shell"),
                false,
            ),
        ];
        for (name, tool_type, server_label, has_handler) in expected_entries {
            let entry = registry
                .lookup(name)
                .unwrap_or_else(|| panic!("missing registry entry '{name}'"));
            assert_eq!(entry.tool_type, tool_type, "unexpected type for '{name}'");
            assert_eq!(
                entry.server_label.as_deref(),
                server_label,
                "unexpected server label for '{name}'"
            );
            assert_eq!(entry.handler.is_some(), has_handler, "unexpected handler for '{name}'");
        }
        assert_eq!(registry.lookup("freeform").unwrap().config["name"], "freeform");
        assert_eq!(registry.lookup("echo").unwrap().config["name"], "echo");
        assert_eq!(
            registry.lookup("mcp__counter__increment").unwrap().config["tool_name"],
            "increment"
        );
        assert_eq!(
            registry.lookup("web_search").unwrap().config["search_context_size"],
            "low"
        );
        assert_eq!(
            registry.lookup("file_search").unwrap().config["vector_store_ids"][0],
            "vs_test"
        );
        assert_eq!(
            registry.lookup("agentic_ns__mcp__shell__run").unwrap().config["tools"][0]["name"],
            "agentic_ns__mcp__shell__run"
        );
        for name in [
            "mcp__counter__increment",
            "mcp__counter__get_value",
            "web_search",
            "file_search",
            "code_interpreter",
        ] {
            assert!(registry.is_gateway_owned_name(name), "'{name}' should be gateway-owned");
        }
        for name in ["echo", "freeform", "agentic_ns__mcp__shell__run"] {
            assert!(!registry.is_gateway_owned_name(name), "'{name}' should be client-owned");
        }

        let ResponsesTool::Mcp(declared) = &tools[1] else {
            panic!("expected MCP declaration");
        };
        assert_eq!(declared.discovered_tools.len(), 2);
        assert_eq!(
            tools[1]
                .to_function_tools()
                .into_iter()
                .map(|tool| tool.name)
                .collect::<Vec<_>>(),
            ["mcp__counter__increment", "mcp__counter__get_value"]
        );

        let ResponsesTool::Namespace(namespace) = &tools[5] else {
            panic!("expected namespace declaration");
        };
        assert!(matches!(
            namespace.tools.as_slice(),
            [crate::types::tools::CodexNamespaceMember::Function(function)] if function.name.as_str() == "run"
        ));
        assert_namespace_call_restoration(&registry);
    }

    #[tokio::test]
    async fn build_with_handlers_retains_mcp_discovery_failure_output() {
        let mut tools = vec![declaration("unreachable")];
        let mut executors = GatewayExecutors::default();

        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("discovery failures should become response metadata");

        let [list_tools] = registry.mcp_list_tools_items() else {
            panic!("expected one MCP list-tools item");
        };
        assert_eq!(list_tools.server_label, "unreachable");
        assert!(list_tools.tools.is_empty());
        assert!(
            list_tools
                .error
                .as_deref()
                .is_some_and(|error| error.contains("failed"))
        );
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn duplicate_mcp_server_labels_are_rejected() {
        let mut tools = vec![declaration("counter"), declaration("counter")];
        let mut executors = GatewayExecutors::default();

        let error = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect_err("duplicate server_label must fail");

        assert!(
            matches!(error, ToolError::Config(message) if message.contains("duplicate MCP declarations") && message.contains("counter"))
        );
    }

    #[tokio::test]
    async fn cross_server_internal_name_collisions_are_rejected() {
        let internal_name = "mcp__foo__bar__baz";
        let mut executors = GatewayExecutors::default();
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "foo".to_owned(),
            handlers: vec![discovered_handler("foo", "bar__baz", internal_name)],
        });
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "foo__bar".to_owned(),
            handlers: vec![discovered_handler("foo__bar", "baz", internal_name)],
        });
        let mut tools = vec![declaration("foo"), declaration("foo__bar")];

        let error = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect_err("colliding derived MCP names must fail");

        assert!(matches!(
            error,
            ToolError::Config(message)
                if message.contains(internal_name) && message.matches("MCP tool").count() == 2
        ));
    }

    #[tokio::test]
    async fn discovered_mcp_name_collision_with_function_is_rejected_in_any_order() {
        let internal_name = "mcp__counter__increment";

        for mcp_first in [false, true] {
            let function = serde_json::from_value(serde_json::json!({
                "type": "function",
                "name": internal_name
            }))
            .expect("function declaration");
            let mcp = declaration("counter");
            let mut tools = if mcp_first {
                vec![mcp, function]
            } else {
                vec![function, mcp]
            };
            let mut executors = GatewayExecutors::default();
            executors.insert(GatewayExecutorRegistration::Mcp {
                server_label: "counter".to_owned(),
                handlers: vec![discovered_handler("counter", "increment", internal_name)],
            });

            let error = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
                .await
                .expect_err("MCP internal name must not overwrite a function");

            assert!(matches!(
                error,
                ToolError::Config(message)
                    if message.contains(internal_name)
                        && message.contains("MCP tool")
                        && message.contains("function tool")
            ));
        }
    }
}
