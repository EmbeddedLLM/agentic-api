use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::Serialize;
use serde_json::Value;

use crate::types::event::MessageStatus;
use crate::types::io::{
    FunctionTool, FunctionToolResultMessage, InputFunctionToolCall, InputItem, InputToolSearchCall, OutputItem,
    ResponsesInput, ToolCallOutput, ToolChoice, ToolSearchCall, ToolSearchOutputMessage,
};
use crate::types::request_response::{RequestPayload, ToolSearchReadiness};
use crate::types::tools::{
    CodexNamespaceMember, CodexNamespaceToolParam, FunctionToolParam, NonEmptyToolName, ResponsesTool,
    ToolSearchExecution, ToolSearchStatus, ToolSearchToolParam,
};
use crate::utils::common::{deserialize_from_str, serialize_to_string, serialize_to_value};

use super::ToolError;
use super::names::validate_model_visible_declared_names;

const SYNTHETIC_TOOL_NAME: &str = "tool_search";

/// Stable public identity used to compare definitions accumulated from search outputs.
///
/// Equality remains type-aware while the state builder also indexes the visible name
/// separately, so returning the same name under a different supported kind is rejected.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum LoadedToolIdentity {
    Function(String),
    Namespace(String),
    Mcp(String),
}

impl LoadedToolIdentity {
    fn name(&self) -> &str {
        match self {
            Self::Function(name) | Self::Namespace(name) | Self::Mcp(name) => name,
        }
    }

    const fn kind(&self) -> &'static str {
        match self {
            Self::Function(_) => "function",
            Self::Namespace(_) => "namespace",
            Self::Mcp(_) => "MCP server",
        }
    }
}

struct DefinitionRecord {
    identity: LoadedToolIdentity,
    canonical: Value,
    public_index: usize,
    loaded: bool,
    namespace_members: Option<NamespaceMemberRecords>,
}

struct NamespaceMemberRecord {
    canonical: Value,
    public_member_index: usize,
    loaded: bool,
}

struct NamespaceMemberRecords {
    ordered: Vec<NamespaceMemberRecord>,
    indexes: HashMap<String, usize>,
    unloaded_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolSearchActivity {
    Inactive,
    Active,
}

struct PendingSearchCall {
    call_id: String,
}

struct DefinitionAccumulator<'a> {
    public_tools: &'a mut Vec<ResponsesTool>,
    definitions: &'a mut Vec<DefinitionRecord>,
    definition_indexes: &'a mut HashMap<String, usize>,
    loaded_public_tools: &'a mut Vec<ResponsesTool>,
    withheld_function_names: &'a mut HashSet<String>,
    mcp_load_positions: &'a mut HashMap<String, usize>,
    trusted_restored_identities: &'a HashSet<LoadedToolIdentity>,
    prior_unknown_namespace_calls: HashMap<String, HashSet<String>>,
    unqualified_call_positions: HashMap<String, usize>,
    current_history_position: Option<usize>,
}

struct DefinitionViews<'a> {
    public_tools: &'a mut Vec<ResponsesTool>,
    definitions: &'a mut Vec<DefinitionRecord>,
    definition_indexes: &'a mut HashMap<String, usize>,
    loaded_public_tools: &'a mut Vec<ResponsesTool>,
    withheld_function_names: &'a mut HashSet<String>,
    mcp_load_positions: &'a mut HashMap<String, usize>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum CatalogEntry {
    Function {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    Namespace {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    Mcp {
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        server_description: Option<String>,
    },
}

impl CatalogEntry {
    fn display_name(&self) -> &str {
        match self {
            Self::Function { name, .. } | Self::Namespace { name, .. } => name,
            Self::Mcp { server_label, .. } => server_label,
        }
    }

    fn description(&self) -> Option<&str> {
        match self {
            Self::Function { description, .. } | Self::Namespace { description, .. } => description.as_deref(),
            Self::Mcp { server_description, .. } => server_description.as_deref(),
        }
    }
}

/// Pure, request-scoped state derived from fully rehydrated public history.
///
/// The state deliberately has no `Serialize` implementation and its `Debug`
/// output contains counts only. Canonical definitions can contain MCP
/// credentials and stay private to equality checks.
pub struct ToolSearchState {
    activity: ToolSearchActivity,
    public_effective_tools: Option<Vec<ResponsesTool>>,
    private_upstream_tools: Option<Vec<ResponsesTool>>,
    private_upstream_input: Option<ResponsesInput>,
    loaded_public_tools: Vec<ResponsesTool>,
    synthetic_function: Option<FunctionTool>,
    withheld_function_names: HashSet<String>,
    mcp_load_positions: HashMap<String, usize>,
    unqualified_call_positions: HashMap<String, usize>,
}

impl fmt::Debug for ToolSearchState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolSearchState")
            .field("activity", &self.activity)
            .field("active", &self.is_active())
            .field(
                "public_effective_tool_count",
                &self.public_effective_tools.as_ref().map_or(0, Vec::len),
            )
            .field(
                "private_upstream_tool_count",
                &self.private_upstream_tools.as_ref().map_or(0, Vec::len),
            )
            .field("loaded_public_tool_count", &self.loaded_public_tools.len())
            .field("has_private_upstream_input", &self.private_upstream_input.is_some())
            .field("has_synthetic_function", &self.synthetic_function.is_some())
            .field("withheld_function_count", &self.withheld_function_names.len())
            .field("mcp_load_count", &self.mcp_load_positions.len())
            .field("unqualified_history_call_count", &self.unqualified_call_positions.len())
            .finish()
    }
}

impl Default for ToolSearchState {
    fn default() -> Self {
        Self {
            activity: ToolSearchActivity::Inactive,
            public_effective_tools: None,
            private_upstream_tools: None,
            private_upstream_input: None,
            loaded_public_tools: Vec::new(),
            synthetic_function: None,
            withheld_function_names: HashSet::new(),
            mcp_load_positions: HashMap::new(),
            unqualified_call_positions: HashMap::new(),
        }
    }
}

impl ToolSearchState {
    /// Build deterministic public/private views from ordered public history.
    ///
    /// This function performs no network, storage, clock, random-ID, or
    /// transport work. It runs in linear time in input items and definitions;
    /// vectors retain declaration/history order and maps are lookup-only.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] for an invalid public declaration,
    /// call/output ordering or linkage error, duplicate/conflicting definition,
    /// or normalized-name collision.
    pub fn build(request: &RequestPayload) -> Result<Self, ToolError> {
        Self::build_with_loaded_tools(request, &[], false)
    }

    /// Build state with public loaded definitions restored from typed response
    /// metadata when compaction has removed the original search pair.
    ///
    /// The restored definitions pass through the same validation and loading
    /// logic as definitions in a public `tool_search_output`.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] under the same conditions as [`Self::build`].
    pub fn build_with_loaded_tools(
        request: &RequestPayload,
        restored_loaded_tools: &[ResponsesTool],
        restore_only_declared: bool,
    ) -> Result<Self, ToolError> {
        let active_input = request.input.model_input();
        let readiness = request.tool_search_readiness_for_input(active_input.as_ref())?;
        if readiness == ToolSearchReadiness::Inactive {
            return Ok(Self::default());
        }

        let input_items = match active_input.as_ref() {
            ResponsesInput::Text(_) => &[][..],
            ResponsesInput::Items(items) => items.as_slice(),
        };
        let has_search_history = input_items
            .iter()
            .any(|item| matches!(item, InputItem::ToolSearchCall(_) | InputItem::ToolSearchOutput(_)));
        let declaration = request
            .tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .find_map(|tool| match tool {
                ResponsesTool::ToolSearch(declaration) => Some(declaration),
                _ => None,
            });
        if declaration.is_none() && !has_search_history {
            return Err(ToolError::Config(
                "defer_loading requires a tool_search declaration or replayed tool-search history".to_owned(),
            ));
        }

        let tools_were_present = request.tools.is_some();
        let mut public_tools = request.tools.clone().unwrap_or_default();
        let mut definitions = Vec::with_capacity(public_tools.len());
        let mut definition_indexes = HashMap::with_capacity(public_tools.len());
        index_initial_definitions(&public_tools, &mut definitions, &mut definition_indexes)?;
        let mut withheld_function_names =
            initial_withheld_function_names(&public_tools, &definitions, &definition_indexes)?;
        let mut mcp_load_positions = HashMap::new();
        let mut unqualified_call_positions = HashMap::new();

        let mut loaded_public_tools = Vec::new();
        let trusted_restored_identities = restore_loaded_definitions(
            restored_loaded_tools,
            DefinitionViews {
                public_tools: &mut public_tools,
                definitions: &mut definitions,
                definition_indexes: &mut definition_indexes,
                loaded_public_tools: &mut loaded_public_tools,
                withheld_function_names: &mut withheld_function_names,
                mcp_load_positions: &mut mcp_load_positions,
            },
            restore_only_declared,
        )?;
        let private_upstream_input = prepare_history(
            active_input.as_ref(),
            DefinitionViews {
                public_tools: &mut public_tools,
                definitions: &mut definitions,
                definition_indexes: &mut definition_indexes,
                loaded_public_tools: &mut loaded_public_tools,
                withheld_function_names: &mut withheld_function_names,
                mcp_load_positions: &mut mcp_load_positions,
            },
            &mut unqualified_call_positions,
            &trusted_restored_identities,
        )?;

        validate_model_visible_declared_names(&public_tools)?;

        let catalog = build_catalog(&public_tools, &definitions, &definition_indexes);
        let synthetic_function = declaration
            .map(|declaration| synthetic_function(declaration, &catalog))
            .transpose()?;
        let private_tools = build_private_tools(
            &public_tools,
            &definitions,
            &definition_indexes,
            synthetic_function.as_ref(),
        );
        let public_effective_tools = (tools_were_present || !public_tools.is_empty()).then_some(public_tools);
        let private_upstream_tools = (tools_were_present || !private_tools.is_empty()).then_some(private_tools);

        Ok(Self {
            activity: ToolSearchActivity::Active,
            public_effective_tools,
            private_upstream_tools,
            private_upstream_input: Some(private_upstream_input),
            loaded_public_tools,
            synthetic_function,
            withheld_function_names,
            mcp_load_positions,
            unqualified_call_positions,
        })
    }

    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self.activity, ToolSearchActivity::Active)
    }

    #[must_use]
    pub fn public_effective_tools(&self) -> Option<&[ResponsesTool]> {
        self.public_effective_tools.as_deref()
    }

    /// Public definitions resolved by completed search outputs, in first-load order.
    ///
    /// This remains separate from `public_effective_tools`: an initially
    /// deferred definition stays deferred publicly even after becoming loaded.
    #[must_use]
    pub fn loaded_public_tools(&self) -> &[ResponsesTool] {
        &self.loaded_public_tools
    }

    /// Private synthetic function declaration used by request-scoped registry and upstream normalization.
    #[must_use]
    pub const fn synthetic_function(&self) -> Option<&FunctionTool> {
        self.synthetic_function.as_ref()
    }

    #[must_use]
    pub(crate) fn withheld_function_names(&self) -> &HashSet<String> {
        &self.withheld_function_names
    }

    #[must_use]
    pub(crate) fn mcp_load_positions(&self) -> &HashMap<String, usize> {
        &self.mcp_load_positions
    }

    #[must_use]
    pub(crate) fn unqualified_call_positions(&self) -> &HashMap<String, usize> {
        &self.unqualified_call_positions
    }

    /// Materialize the private inference request for function, namespace, and
    /// MCP tools from views prepared in the single state-building pass. The
    /// public request is borrowed and never mutated.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] when the effective tool choice conflicts
    /// with the prepared private tool set.
    pub fn private_inference_request(&self, public: &RequestPayload) -> Result<RequestPayload, ToolError> {
        validate_effective_tool_choice(public.tool_choice.as_ref(), &self.withheld_function_names)?;
        let mut private = public.clone();
        if let Some(input) = &self.private_upstream_input {
            private.input.clone_from(input);
        }
        private.tools.clone_from(&self.private_upstream_tools);
        Ok(private)
    }
}

fn validate_effective_tool_choice(
    tool_choice: Option<&ToolChoice>,
    withheld_function_names: &HashSet<String>,
) -> Result<(), ToolError> {
    let targets_withheld = match tool_choice {
        Some(ToolChoice::Function { namespace, name }) => {
            let model_name = namespace.as_deref().map_or_else(
                || name.as_str().to_owned(),
                |namespace| super::model_visible_namespace_member_name(namespace, name.as_str()),
            );
            withheld_function_names.contains(&model_name)
        }
        Some(ToolChoice::AllowedTools { tools, .. }) => tools
            .iter()
            .any(|tool| tool.type_.as_str() == "function" && withheld_function_names.contains(tool.name.as_str())),
        _ => false,
    };
    if targets_withheld {
        return Err(ToolError::Config(
            "tool_choice targets a function before its definition is loaded".to_owned(),
        ));
    }
    Ok(())
}

fn restore_loaded_definitions(
    restored_loaded_tools: &[ResponsesTool],
    views: DefinitionViews<'_>,
    restore_only_declared: bool,
) -> Result<HashSet<LoadedToolIdentity>, ToolError> {
    let DefinitionViews {
        public_tools,
        definitions,
        definition_indexes,
        loaded_public_tools,
        withheld_function_names,
        mcp_load_positions,
    } = views;
    let no_trusted_restored_identities = HashSet::new();
    let mut trusted_restored_identities = HashSet::with_capacity(restored_loaded_tools.len());
    let mut accumulator = DefinitionAccumulator {
        public_tools,
        definitions,
        definition_indexes,
        loaded_public_tools,
        withheld_function_names,
        mcp_load_positions,
        trusted_restored_identities: &no_trusted_restored_identities,
        prior_unknown_namespace_calls: HashMap::new(),
        unqualified_call_positions: HashMap::new(),
        current_history_position: None,
    };
    for tool in restored_loaded_tools {
        let Some(tool) = restored_definition_for_load(
            tool,
            accumulator.public_tools,
            accumulator.definitions,
            accumulator.definition_indexes,
            restore_only_declared,
        )?
        else {
            continue;
        };
        load_definition(&tool, &mut accumulator)?;
        let identity = loaded_tool_identity(&tool)?.ok_or_else(|| {
            ToolError::Config("stored tool-search availability contains an unsupported definition".to_owned())
        })?;
        trusted_restored_identities.insert(identity);
    }
    Ok(trusted_restored_identities)
}

fn restored_definition_for_load(
    restored: &ResponsesTool,
    public_tools: &[ResponsesTool],
    definitions: &[DefinitionRecord],
    definition_indexes: &HashMap<String, usize>,
    restore_only_declared: bool,
) -> Result<Option<ResponsesTool>, ToolError> {
    let identity = loaded_tool_identity(restored)?.ok_or_else(|| {
        ToolError::Config("stored tool-search availability contains an unsupported definition".to_owned())
    })?;
    let Some(record) = definition_indexes
        .get(identity.name())
        .and_then(|index| definitions.get(*index))
    else {
        return Ok((!restore_only_declared).then(|| restored.clone()));
    };
    if record.identity != identity {
        return Ok(Some(restored.clone()));
    }
    let declared = &public_tools[record.public_index];
    if matches!((restored, declared), (ResponsesTool::Mcp(_), ResponsesTool::Mcp(_))) {
        let mut sanitized = declared.clone();
        sanitized.sanitize_for_persistence();
        if canonical_definition(&sanitized)? != canonical_definition(restored)? {
            return Err(ToolError::Config(format!(
                "loaded definition for identity '{}' conflicts with its existing type, schema, description, or configuration",
                identity.name()
            )));
        }
        return Ok(Some(declared.clone()));
    }
    Ok(Some(restored.clone()))
}

pub(crate) fn public_item_id(item_id: &str) -> String {
    if item_id.strip_prefix("tsc_").is_some_and(|suffix| !suffix.is_empty()) {
        return item_id.to_owned();
    }
    if let Some(suffix) = item_id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("tsc_{suffix}");
    }
    let domain_separated = format!("tool_search_item:{item_id}");
    format!("tsc_{:016x}", stable_hash(&domain_separated))
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub(crate) fn public_output_item(item_id: &str, call_id: &str, arguments: &str) -> Result<OutputItem, ToolError> {
    if item_id.trim().is_empty() || call_id.trim().is_empty() {
        return Err(invalid_upstream_search_call());
    }
    let arguments = deserialize_from_str::<Value>(arguments)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .ok_or_else(invalid_upstream_search_call)?;
    Ok(OutputItem::ToolSearchCall(ToolSearchCall {
        id: public_item_id(item_id),
        call_id: call_id.to_owned(),
        execution: ToolSearchExecution::Client,
        arguments,
        status: ToolSearchStatus::Completed,
    }))
}

pub(crate) fn public_output_item_from_raw(object: &serde_json::Map<String, Value>) -> Result<OutputItem, ToolError> {
    if object.get("type").and_then(Value::as_str) != Some("function_call")
        || object.get("name").and_then(Value::as_str) != Some(SYNTHETIC_TOOL_NAME)
        || object.get("status").and_then(Value::as_str) != Some("completed")
        || object.get("namespace").is_some_and(|namespace| !namespace.is_null())
    {
        return Err(invalid_upstream_search_call());
    }
    let item_id = required_non_blank_string(object.get("id"))?;
    let call_id = required_non_blank_string(object.get("call_id"))?;
    let arguments = required_non_blank_string(object.get("arguments"))?;
    public_output_item(item_id, call_id, arguments)
}

pub(crate) fn public_added_item(item_id: &str, call_id: &str) -> Result<Value, ToolError> {
    if item_id.trim().is_empty() || call_id.trim().is_empty() {
        return Err(invalid_upstream_search_call());
    }
    Ok(serde_json::json!({
        "id": public_item_id(item_id),
        "type": "tool_search_call",
        "status": "in_progress",
        "arguments": {},
        "call_id": call_id,
        "execution": "client",
    }))
}

fn required_non_blank_string(value: Option<&Value>) -> Result<&str, ToolError> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(invalid_upstream_search_call)
}

pub(crate) fn invalid_upstream_search_call() -> ToolError {
    ToolError::Execution("upstream returned an invalid tool-search call".to_owned())
}

pub(crate) fn invalid_upstream_withheld_function_call() -> ToolError {
    ToolError::Execution("upstream returned a call for a function that has not been loaded".to_owned())
}

fn index_initial_definitions(
    tools: &[ResponsesTool],
    definitions: &mut Vec<DefinitionRecord>,
    definition_indexes: &mut HashMap<String, usize>,
) -> Result<(), ToolError> {
    for (public_index, tool) in tools.iter().enumerate() {
        let Some(identity) = loaded_tool_identity(tool)? else {
            continue;
        };
        if definition_indexes.contains_key(identity.name()) {
            return Err(ToolError::Config(format!(
                "duplicate tool-search definition identity '{}'",
                identity.name()
            )));
        }
        let index = definitions.len();
        definition_indexes.insert(identity.name().to_owned(), index);
        definitions.push(definition_record(tool, identity, public_index, false)?);
    }
    Ok(())
}

fn definition_record(
    tool: &ResponsesTool,
    identity: LoadedToolIdentity,
    public_index: usize,
    dynamically_loaded: bool,
) -> Result<DefinitionRecord, ToolError> {
    let namespace_members = match tool {
        ResponsesTool::Namespace(namespace) => Some(namespace_member_records(namespace, dynamically_loaded)?),
        ResponsesTool::Function(_) | ResponsesTool::Mcp(_) => None,
        ResponsesTool::ToolSearch(_)
        | ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Custom(_)
        | ResponsesTool::Unknown => {
            return Err(ToolError::Config(
                "tool-search definition record received an unsupported tool".to_owned(),
            ));
        }
    };
    let loaded = namespace_members
        .as_ref()
        .map_or(dynamically_loaded, |members| members.unloaded_count == 0);
    Ok(DefinitionRecord {
        identity,
        canonical: canonical_definition(tool)?,
        public_index,
        loaded,
        namespace_members,
    })
}

fn namespace_member_records(
    namespace: &CodexNamespaceToolParam,
    dynamically_loaded: bool,
) -> Result<NamespaceMemberRecords, ToolError> {
    if namespace.tools.is_empty() {
        return Err(ToolError::Config(
            "tool-search namespaces must contain at least one function member".to_owned(),
        ));
    }
    let mut ordered = Vec::with_capacity(namespace.tools.len());
    let mut indexes = HashMap::with_capacity(namespace.tools.len());
    let mut unloaded_count = 0;
    for (public_member_index, member) in namespace.tools.iter().enumerate() {
        let CodexNamespaceMember::Function(function) = member else {
            return Err(ToolError::Config(
                "tool-search namespaces may contain only function members".to_owned(),
            ));
        };
        let name = function.name.as_str();
        if indexes.insert(name.to_owned(), ordered.len()).is_some() {
            return Err(ToolError::Config(format!(
                "duplicate namespace member identity '{}.{name}'",
                namespace.name
            )));
        }
        let loaded = dynamically_loaded || function.defer_loading != Some(true);
        unloaded_count += usize::from(!loaded);
        ordered.push(NamespaceMemberRecord {
            canonical: canonical_namespace_member(function)?,
            public_member_index,
            loaded,
        });
    }
    Ok(NamespaceMemberRecords {
        ordered,
        indexes,
        unloaded_count,
    })
}

fn initial_withheld_function_names(
    public_tools: &[ResponsesTool],
    definitions: &[DefinitionRecord],
    definition_indexes: &HashMap<String, usize>,
) -> Result<HashSet<String>, ToolError> {
    let mut withheld = HashSet::new();
    for tool in public_tools {
        if let ResponsesTool::Function(function) = tool {
            if function.defer_loading == Some(true) {
                withheld.insert(function.name.as_str().to_owned());
            }
            continue;
        }
        let ResponsesTool::Namespace(namespace) = tool else {
            continue;
        };
        let record = definition_indexes
            .get(&namespace.name)
            .and_then(|index| definitions.get(*index))
            .ok_or_else(|| ToolError::Config("namespace availability state is inconsistent".to_owned()))?;
        let members = record
            .namespace_members
            .as_ref()
            .ok_or_else(|| ToolError::Config("namespace availability state is inconsistent".to_owned()))?;
        for member in &namespace.tools {
            let CodexNamespaceMember::Function(function) = member else {
                continue;
            };
            let member_index = members
                .indexes
                .get(function.name.as_str())
                .ok_or_else(|| ToolError::Config("namespace availability state is inconsistent".to_owned()))?;
            let is_withheld = !members.ordered[*member_index].loaded;
            if is_withheld {
                withheld.insert(super::model_visible_namespace_member_name(
                    &namespace.name,
                    function.name.as_str(),
                ));
            }
        }
    }
    Ok(withheld)
}

#[derive(Serialize)]
struct CanonicalToolSearchOutput<'a> {
    tools: &'a [ModelVisibleLoadedTool<'a>],
}

/// Typed model-output projection, deliberately separate from the raw
/// credential-sensitive definition retained for equality and later execution.
#[derive(Serialize)]
#[serde(untagged)]
enum ModelVisibleLoadedTool<'a> {
    Function(ModelVisibleFunction<'a>),
    Namespace(ModelVisibleNamespace<'a>),
    Mcp(ModelVisibleMcp<'a>),
}

#[derive(Serialize)]
struct ModelVisibleFunction<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    #[serde(flatten)]
    definition: &'a FunctionToolParam,
}

#[derive(Serialize)]
struct ModelVisibleNamespace<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
}

#[derive(Serialize)]
struct ModelVisibleMcp<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    server_label: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    server_description: Option<&'a str>,
}

fn prepare_history(
    input: &ResponsesInput,
    views: DefinitionViews<'_>,
    unqualified_call_positions: &mut HashMap<String, usize>,
    trusted_restored_identities: &HashSet<LoadedToolIdentity>,
) -> Result<ResponsesInput, ToolError> {
    let ResponsesInput::Items(items) = input else {
        return Ok(input.clone());
    };
    let mut private_items = Vec::with_capacity(items.len());
    let mut unresolved_call: Option<PendingSearchCall> = None;
    let mut completed_call_ids = HashSet::new();
    let mut item_ids = HashSet::new();
    let DefinitionViews {
        public_tools,
        definitions,
        definition_indexes,
        loaded_public_tools,
        withheld_function_names,
        mcp_load_positions,
    } = views;
    let mut definition_accumulator = DefinitionAccumulator {
        public_tools,
        definitions,
        definition_indexes,
        loaded_public_tools,
        withheld_function_names,
        mcp_load_positions,
        trusted_restored_identities,
        prior_unknown_namespace_calls: HashMap::new(),
        unqualified_call_positions: std::mem::take(unqualified_call_positions),
        current_history_position: None,
    };

    for (position, item) in items.iter().enumerate() {
        definition_accumulator.current_history_position = Some(position);
        match item {
            InputItem::ToolSearchCall(call) => {
                private_items.push(prepare_search_call(
                    call,
                    &mut unresolved_call,
                    &completed_call_ids,
                    &mut item_ids,
                )?);
            }
            InputItem::ToolSearchOutput(output) => {
                private_items.push(prepare_search_output(
                    output,
                    &mut unresolved_call,
                    &mut completed_call_ids,
                    &mut definition_accumulator,
                )?);
            }
            InputItem::FunctionCall(call) => {
                ensure_history_call_is_available(call, &mut definition_accumulator)?;
                private_items.push(item.clone());
            }
            InputItem::Message(_)
            | InputItem::FunctionCallOutput(_)
            | InputItem::CustomToolCall(_)
            | InputItem::CustomToolCallOutput(_)
            | InputItem::Reasoning(_)
            | InputItem::Compaction(_)
            | InputItem::Unknown => private_items.push(item.clone()),
        }
    }

    if unresolved_call.is_some() {
        return Err(ToolError::Config(
            "unresolved tool_search_call requires a matching completed tool_search_output".to_owned(),
        ));
    }
    *unqualified_call_positions = definition_accumulator.unqualified_call_positions;
    Ok(ResponsesInput::Items(private_items))
}

fn ensure_history_call_is_available(
    call: &InputFunctionToolCall,
    definitions: &mut DefinitionAccumulator<'_>,
) -> Result<(), ToolError> {
    if let Some(namespace) = call.namespace.as_deref() {
        let member = definitions
            .definition_indexes
            .get(namespace)
            .and_then(|index| definitions.definitions.get(*index))
            .filter(|record| matches!(record.identity, LoadedToolIdentity::Namespace(_)))
            .and_then(|record| record.namespace_members.as_ref())
            .and_then(|members| members.indexes.get(&call.name).map(|index| &members.ordered[*index]));
        match member {
            Some(member) if !member.loaded => return Err(withheld_function_history_call()),
            Some(_) => {}
            None => {
                definitions
                    .prior_unknown_namespace_calls
                    .entry(namespace.to_owned())
                    .or_default()
                    .insert(call.name.clone());
            }
        }
    } else {
        if definitions.withheld_function_names.contains(&call.name) {
            return Err(withheld_function_history_call());
        }
        let position = definitions
            .current_history_position
            .ok_or_else(|| ToolError::Config("tool-search history position is unavailable".to_owned()))?;
        definitions
            .unqualified_call_positions
            .entry(call.name.clone())
            .or_insert(position);
    }
    Ok(())
}

fn withheld_function_history_call() -> ToolError {
    ToolError::Config("request history calls a function before its definition is loaded".to_owned())
}

fn prepare_search_call(
    call: &InputToolSearchCall,
    unresolved_call: &mut Option<PendingSearchCall>,
    completed_call_ids: &HashSet<String>,
    item_ids: &mut HashSet<String>,
) -> Result<InputItem, ToolError> {
    if call.id.trim().is_empty() {
        return Err(ToolError::Config("tool_search_call id must not be blank".to_owned()));
    }
    if call.call_id.trim().is_empty() {
        return Err(ToolError::Config(
            "tool_search_call call_id must not be blank".to_owned(),
        ));
    }
    if !item_ids.insert(call.id.clone()) {
        return Err(ToolError::Config("duplicate tool_search_call item id".to_owned()));
    }
    if unresolved_call.is_some() {
        return Err(ToolError::Config(
            "ambiguous tool-search history contains a call before the preceding call is resolved".to_owned(),
        ));
    }
    if completed_call_ids.contains(call.call_id.as_str()) {
        return Err(ToolError::Config("duplicate tool_search_call call_id".to_owned()));
    }
    let canonical_arguments = serialize_to_string(&Value::Object(call.arguments.clone()))
        .map_err(|_| ToolError::Config("tool_search_call arguments could not be canonicalized safely".to_owned()))?;
    *unresolved_call = Some(PendingSearchCall {
        call_id: call.call_id.clone(),
    });
    Ok(InputItem::FunctionCall(InputFunctionToolCall {
        id: Some(call.id.clone()),
        call_id: call.call_id.clone(),
        name: SYNTHETIC_TOOL_NAME.to_owned(),
        namespace: None,
        arguments: canonical_arguments,
        status: Some(MessageStatus::Completed),
    }))
}

fn prepare_search_output(
    output: &ToolSearchOutputMessage,
    unresolved_call: &mut Option<PendingSearchCall>,
    completed_call_ids: &mut HashSet<String>,
    definition_accumulator: &mut DefinitionAccumulator<'_>,
) -> Result<InputItem, ToolError> {
    if output.call_id.trim().is_empty() {
        return Err(ToolError::Config(
            "tool_search_output call_id must not be blank".to_owned(),
        ));
    }
    if completed_call_ids.contains(output.call_id.as_str()) {
        return Err(ToolError::Config("duplicate tool_search_output call_id".to_owned()));
    }
    let Some(pending) = unresolved_call.take() else {
        return Err(ToolError::Config(
            "orphan tool_search_output has no unresolved call".to_owned(),
        ));
    };
    if pending.call_id != output.call_id {
        return Err(ToolError::Config(
            "tool_search_output call_id does not match the preceding unresolved call".to_owned(),
        ));
    }
    for tool in &output.tools {
        load_definition(tool, definition_accumulator)?;
    }
    let projected_tools = model_visible_output_tools(&output.tools)?;
    let canonical_value = serialize_to_value(&CanonicalToolSearchOutput {
        tools: &projected_tools,
    })
    .map_err(|_| ToolError::Config("tool_search_output could not be canonicalized safely".to_owned()))?;
    let canonical_output = serialize_to_string(&canonical_value)
        .map_err(|_| ToolError::Config("tool_search_output could not be canonicalized safely".to_owned()))?;
    completed_call_ids.insert(output.call_id.clone());
    Ok(InputItem::FunctionCallOutput(FunctionToolResultMessage {
        call_id: output.call_id.clone(),
        output: ToolCallOutput::Text(canonical_output),
    }))
}

fn model_visible_output_tools(tools: &[ResponsesTool]) -> Result<Vec<ModelVisibleLoadedTool<'_>>, ToolError> {
    tools
        .iter()
        .map(|tool| match tool {
            ResponsesTool::Function(definition) => Ok(ModelVisibleLoadedTool::Function(ModelVisibleFunction {
                type_: "function",
                definition,
            })),
            ResponsesTool::Namespace(namespace) => Ok(ModelVisibleLoadedTool::Namespace(ModelVisibleNamespace {
                type_: "namespace",
                name: &namespace.name,
                description: namespace.description.as_deref(),
            })),
            ResponsesTool::Mcp(mcp) => Ok(ModelVisibleLoadedTool::Mcp(ModelVisibleMcp {
                type_: "mcp",
                server_label: &mcp.server_label,
                server_description: mcp.server_description.as_deref(),
            })),
            ResponsesTool::ToolSearch(_)
            | ResponsesTool::WebSearch(_)
            | ResponsesTool::FileSearch(_)
            | ResponsesTool::CodeInterpreter(_)
            | ResponsesTool::Custom(_)
            | ResponsesTool::Unknown => Err(ToolError::Config(
                "tool_search_output contains an unsupported model-output definition".to_owned(),
            )),
        })
        .collect()
}

fn load_definition(tool: &ResponsesTool, definitions: &mut DefinitionAccumulator<'_>) -> Result<(), ToolError> {
    let identity = loaded_tool_identity(tool)?
        .ok_or_else(|| ToolError::Config("tool_search_output contains an unsupported tool definition".to_owned()))?;
    let canonical = canonical_definition(tool)?;
    if let Some(index) = definitions.definition_indexes.get(identity.name()).copied() {
        let record = &mut definitions.definitions[index];
        if record.identity != identity || record.canonical != canonical {
            if record.identity == identity
                && matches!(tool, ResponsesTool::Mcp(_))
                && definitions.trusted_restored_identities.contains(&identity)
            {
                let mut sanitized = definitions.public_tools[record.public_index].clone();
                sanitized.sanitize_for_persistence();
                if canonical_definition(&sanitized)? == canonical {
                    return Ok(());
                }
            }
            return Err(ToolError::Config(format!(
                "loaded definition for identity '{}' conflicts with its existing type, schema, description, or configuration",
                identity.name()
            )));
        }
        if let ResponsesTool::Namespace(returned) = tool {
            return load_namespace_members(
                returned,
                record,
                definitions.public_tools,
                definitions.loaded_public_tools,
                definitions.withheld_function_names,
                &definitions.prior_unknown_namespace_calls,
                &definitions.unqualified_call_positions,
            );
        }
        if !record.loaded {
            if definitions.withheld_function_names.contains(record.identity.name())
                && definitions
                    .unqualified_call_positions
                    .contains_key(record.identity.name())
            {
                return Err(withheld_function_history_call());
            }
            let deferred_mcp = matches!(
                &definitions.public_tools[record.public_index],
                ResponsesTool::Mcp(mcp) if mcp.defer_loading == Some(true)
            );
            if let LoadedToolIdentity::Mcp(server_label) = &record.identity
                && deferred_mcp
                && let Some(position) = definitions.current_history_position
            {
                definitions
                    .mcp_load_positions
                    .entry(server_label.clone())
                    .or_insert(position);
            }
            record.loaded = true;
            definitions.withheld_function_names.remove(record.identity.name());
            definitions
                .loaded_public_tools
                .push(definitions.public_tools[record.public_index].clone());
        }
        return Ok(());
    }

    match tool {
        ResponsesTool::Function(function)
            if definitions
                .unqualified_call_positions
                .contains_key(function.name.as_str()) =>
        {
            return Err(withheld_function_history_call());
        }
        ResponsesTool::Namespace(namespace) => ensure_namespace_members_do_not_resolve_prior_calls(
            namespace,
            namespace.tools.iter(),
            &definitions.prior_unknown_namespace_calls,
            &definitions.unqualified_call_positions,
        )?,
        ResponsesTool::Mcp(mcp) => {
            if let Some(position) = definitions.current_history_position {
                definitions
                    .mcp_load_positions
                    .entry(mcp.server_label.clone())
                    .or_insert(position);
            }
        }
        _ => {}
    }
    let public_index = definitions.public_tools.len();
    definitions.public_tools.push(tool.clone());
    let index = definitions.definitions.len();
    definitions.definition_indexes.insert(identity.name().to_owned(), index);
    definitions
        .definitions
        .push(definition_record(tool, identity, public_index, true)?);
    definitions.loaded_public_tools.push(tool.clone());
    Ok(())
}

fn ensure_namespace_members_do_not_resolve_prior_calls<'a>(
    namespace: &CodexNamespaceToolParam,
    members: impl Iterator<Item = &'a CodexNamespaceMember>,
    prior_unknown_namespace_calls: &HashMap<String, HashSet<String>>,
    unqualified_call_positions: &HashMap<String, usize>,
) -> Result<(), ToolError> {
    let prior_public_members = prior_unknown_namespace_calls.get(&namespace.name);
    for member in members {
        let CodexNamespaceMember::Function(function) = member else {
            continue;
        };
        let public_match = prior_public_members.is_some_and(|members| members.contains(function.name.as_str()));
        let flat_name = super::model_visible_namespace_member_name(&namespace.name, function.name.as_str());
        if public_match || unqualified_call_positions.contains_key(&flat_name) {
            return Err(withheld_function_history_call());
        }
    }
    Ok(())
}

fn load_namespace_members(
    returned: &CodexNamespaceToolParam,
    record: &mut DefinitionRecord,
    public_tools: &mut [ResponsesTool],
    loaded_public_tools: &mut Vec<ResponsesTool>,
    withheld_function_names: &mut HashSet<String>,
    prior_unknown_namespace_calls: &HashMap<String, HashSet<String>>,
    unqualified_call_positions: &HashMap<String, usize>,
) -> Result<(), ToolError> {
    if returned.tools.is_empty() {
        return Err(ToolError::Config(
            "tool_search_output namespaces must contain at least one function member".to_owned(),
        ));
    }
    let ResponsesTool::Namespace(public_namespace) = &mut public_tools[record.public_index] else {
        return Err(ToolError::Config(
            "namespace identity conflicts with an existing non-namespace definition".to_owned(),
        ));
    };
    let members = record
        .namespace_members
        .as_mut()
        .ok_or_else(|| ToolError::Config("namespace definition is missing prepared member state".to_owned()))?;
    ensure_namespace_members_do_not_resolve_prior_calls(
        returned,
        returned.tools.iter().filter(|member| match member {
            CodexNamespaceMember::Function(function) => !members.indexes.contains_key(function.name.as_str()),
            CodexNamespaceMember::Unknown => true,
        }),
        prior_unknown_namespace_calls,
        unqualified_call_positions,
    )?;
    let mut newly_loaded = Vec::new();
    for member in &returned.tools {
        let CodexNamespaceMember::Function(returned_function) = member else {
            return Err(ToolError::Config(
                "tool_search_output namespaces may contain only function members".to_owned(),
            ));
        };
        let member_name = returned_function.name.as_str();
        let canonical = canonical_namespace_member(returned_function)?;
        if let Some(member_index) = members.indexes.get(member_name).copied() {
            let member_record = &mut members.ordered[member_index];
            if member_record.canonical != canonical {
                return Err(ToolError::Config(format!(
                    "loaded namespace member '{}.{member_name}' conflicts with its existing schema, description, or configuration",
                    returned.name
                )));
            }
            if !member_record.loaded {
                let unloaded_count = members.unloaded_count.checked_sub(1).ok_or_else(|| {
                    ToolError::Config("namespace member availability state is inconsistent".to_owned())
                })?;
                member_record.loaded = true;
                members.unloaded_count = unloaded_count;
                withheld_function_names
                    .remove(&super::model_visible_namespace_member_name(&returned.name, member_name));
                newly_loaded.push(public_namespace.tools[member_record.public_member_index].clone());
            }
            continue;
        }

        let public_member_index = public_namespace.tools.len();
        public_namespace.tools.push(member.clone());
        members.indexes.insert(member_name.to_owned(), members.ordered.len());
        members.ordered.push(NamespaceMemberRecord {
            canonical,
            public_member_index,
            loaded: true,
        });
        newly_loaded.push(member.clone());
    }
    if !newly_loaded.is_empty() {
        let mut loaded_subset = public_namespace.clone();
        loaded_subset.tools = newly_loaded;
        loaded_public_tools.push(ResponsesTool::Namespace(loaded_subset));
    }
    record.loaded = members.unloaded_count == 0;
    Ok(())
}

fn loaded_tool_identity(tool: &ResponsesTool) -> Result<Option<LoadedToolIdentity>, ToolError> {
    let identity = match tool {
        ResponsesTool::Function(function) => LoadedToolIdentity::Function(function.name.as_str().to_owned()),
        ResponsesTool::Namespace(namespace) => LoadedToolIdentity::Namespace(namespace.name.clone()),
        ResponsesTool::Mcp(mcp) => LoadedToolIdentity::Mcp(mcp.server_label.clone()),
        ResponsesTool::ToolSearch(_)
        | ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Custom(_)
        | ResponsesTool::Unknown => return Ok(None),
    };
    if identity.name().trim().is_empty() {
        return Err(ToolError::Config(format!(
            "{} definition identity must not be blank",
            identity.kind()
        )));
    }
    if matches!(
        &identity,
        LoadedToolIdentity::Function(name) | LoadedToolIdentity::Namespace(name)
            if name == SYNTHETIC_TOOL_NAME
    ) {
        return Err(ToolError::Config(
            "model-visible tool name 'tool_search' is reserved while tool search is active".to_owned(),
        ));
    }
    Ok(Some(identity))
}

fn canonical_definition(tool: &ResponsesTool) -> Result<Value, ToolError> {
    let projected = match tool {
        ResponsesTool::Namespace(namespace) => {
            let mut namespace = namespace.clone();
            namespace.tools.clear();
            ResponsesTool::Namespace(namespace)
        }
        other => other.clone(),
    };
    serialize_to_value(&projected)
        .map_err(|_| ToolError::Config("tool-search definition could not be compared safely".to_owned()))
}

fn canonical_namespace_member(function: &FunctionToolParam) -> Result<Value, ToolError> {
    serialize_to_value(function)
        .map_err(|_| ToolError::Config("namespace member definition could not be compared safely".to_owned()))
}

fn build_catalog(
    public_tools: &[ResponsesTool],
    definitions: &[DefinitionRecord],
    definition_indexes: &HashMap<String, usize>,
) -> Vec<CatalogEntry> {
    public_tools
        .iter()
        .filter_map(|tool| {
            let identity = loaded_tool_identity(tool).ok().flatten()?;
            let record = &definitions[*definition_indexes.get(identity.name())?];
            if record.loaded {
                return None;
            }
            match tool {
                ResponsesTool::Function(function) if function.defer_loading == Some(true) => {
                    Some(CatalogEntry::Function {
                        name: function.name.as_str().to_owned(),
                        description: function.description.clone(),
                    })
                }
                ResponsesTool::Namespace(namespace)
                    if namespace_has_withheld_member(record.namespace_members.as_ref()) =>
                {
                    Some(CatalogEntry::Namespace {
                        name: namespace.name.clone(),
                        description: namespace.description.clone(),
                    })
                }
                ResponsesTool::Mcp(mcp) if mcp.defer_loading == Some(true) => Some(CatalogEntry::Mcp {
                    server_label: mcp.server_label.clone(),
                    server_description: mcp.server_description.clone(),
                }),
                ResponsesTool::Function(_)
                | ResponsesTool::Namespace(_)
                | ResponsesTool::Mcp(_)
                | ResponsesTool::ToolSearch(_)
                | ResponsesTool::WebSearch(_)
                | ResponsesTool::FileSearch(_)
                | ResponsesTool::CodeInterpreter(_)
                | ResponsesTool::Custom(_)
                | ResponsesTool::Unknown => None,
            }
        })
        .collect()
}

/// Catalog prose deliberately follows the provider-characterization shape:
/// declaration text, then one ordered semicolon-delimited list of `name —
/// description` entries. It never uses schemas or execution configuration.
fn synthetic_description(declaration: &ToolSearchToolParam, catalog: &[CatalogEntry]) -> String {
    if catalog.is_empty() {
        return declaration.description.clone();
    }
    let entries = catalog
        .iter()
        .map(|entry| {
            entry.description().map_or_else(
                || entry.display_name().to_owned(),
                |description| {
                    let description = description.trim();
                    if description.is_empty() {
                        entry.display_name().to_owned()
                    } else {
                        format!("{} — {description}", entry.display_name())
                    }
                },
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let noun = if catalog.len() == 1 { "entry" } else { "entries" };
    format!(
        "{}. Available catalog {noun}: {entries}.",
        declaration.description.trim().trim_end_matches('.')
    )
}

fn synthetic_function(declaration: &ToolSearchToolParam, catalog: &[CatalogEntry]) -> Result<FunctionTool, ToolError> {
    let name = NonEmptyToolName::try_from(SYNTHETIC_TOOL_NAME)
        .map_err(|_| ToolError::Config("reserved synthetic tool name is invalid".to_owned()))?;
    let param = FunctionToolParam {
        name,
        description: Some(synthetic_description(declaration, catalog)),
        parameters: Some(Value::Object(declaration.parameters.clone())),
        strict: Some(true),
        defer_loading: None,
        extra: HashMap::new(),
    };
    Ok(FunctionTool::from(&param))
}

fn build_private_tools(
    public_tools: &[ResponsesTool],
    definitions: &[DefinitionRecord],
    definition_indexes: &HashMap<String, usize>,
    synthetic_function: Option<&FunctionTool>,
) -> Vec<ResponsesTool> {
    public_tools
        .iter()
        .filter_map(|tool| match tool {
            ResponsesTool::ToolSearch(_) => synthetic_function.map(function_tool_as_response),
            ResponsesTool::Function(_) | ResponsesTool::Namespace(_) | ResponsesTool::Mcp(_) => {
                private_definition(tool, definitions, definition_indexes)
            }
            ResponsesTool::WebSearch(_)
            | ResponsesTool::FileSearch(_)
            | ResponsesTool::CodeInterpreter(_)
            | ResponsesTool::Custom(_)
            | ResponsesTool::Unknown => Some(tool.clone()),
        })
        .collect()
}

fn private_definition(
    tool: &ResponsesTool,
    definitions: &[DefinitionRecord],
    definition_indexes: &HashMap<String, usize>,
) -> Option<ResponsesTool> {
    let identity = loaded_tool_identity(tool).ok().flatten()?;
    let loaded = definitions[*definition_indexes.get(identity.name())?].loaded;
    match tool {
        ResponsesTool::Function(function) if loaded || function.defer_loading != Some(true) => {
            let mut function = function.clone();
            function.defer_loading = None;
            Some(ResponsesTool::Function(function))
        }
        ResponsesTool::Namespace(namespace) => private_namespace(
            namespace,
            definitions[*definition_indexes.get(identity.name())?]
                .namespace_members
                .as_ref(),
        )
        .map(ResponsesTool::Namespace),
        ResponsesTool::Mcp(mcp) if loaded || mcp.defer_loading != Some(true) => {
            let mut mcp = mcp.clone();
            mcp.defer_loading = None;
            Some(ResponsesTool::Mcp(mcp))
        }
        ResponsesTool::Function(_)
        | ResponsesTool::Mcp(_)
        | ResponsesTool::ToolSearch(_)
        | ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Custom(_)
        | ResponsesTool::Unknown => None,
    }
}

fn private_namespace(
    namespace: &CodexNamespaceToolParam,
    member_records: Option<&NamespaceMemberRecords>,
) -> Option<CodexNamespaceToolParam> {
    let member_records = member_records?;
    let tools = namespace
        .tools
        .iter()
        .filter_map(|member| match member {
            CodexNamespaceMember::Function(function)
                if member_records
                    .indexes
                    .get(function.name.as_str())
                    .is_some_and(|index| member_records.ordered[*index].loaded) =>
            {
                let mut function = function.clone();
                function.defer_loading = None;
                Some(CodexNamespaceMember::Function(function))
            }
            CodexNamespaceMember::Function(_) => None,
            CodexNamespaceMember::Unknown => Some(CodexNamespaceMember::Unknown),
        })
        .collect::<Vec<_>>();
    (!tools.is_empty()).then(|| CodexNamespaceToolParam {
        tools,
        ..namespace.clone()
    })
}

fn namespace_has_withheld_member(member_records: Option<&NamespaceMemberRecords>) -> bool {
    member_records.is_some_and(|members| members.unloaded_count != 0)
}

fn function_tool_as_response(function: &FunctionTool) -> ResponsesTool {
    ResponsesTool::Function(FunctionToolParam {
        name: NonEmptyToolName::try_from(function.name.as_str())
            .expect("the synthetic function uses a fixed non-empty name"),
        description: function.description.clone(),
        parameters: function.parameters.clone(),
        strict: function.strict,
        defer_loading: None,
        extra: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tools::McpDiscoveredToolParam;

    #[test]
    fn public_tool_search_item_ids_are_stable_and_domain_separated() {
        assert_eq!(public_item_id("tsc_existing"), "tsc_existing");
        assert_eq!(public_item_id("fc_search_1"), "tsc_search_1");
        let first = public_item_id("provider-item-1");
        assert_eq!(first, public_item_id("provider-item-1"));
        assert!(first.starts_with("tsc_"));
        assert_ne!(first, crate::tool::custom::public_item_id("provider-item-1"));
    }

    #[test]
    fn internal_discovered_mcp_details_never_enter_model_visible_pair_projection() {
        let mut request: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "store": false,
            "parallel_tool_calls": false,
            "tools": [{
                "type": "tool_search",
                "execution": "client",
                "description": "Find a tool",
                "parameters": {"type": "object"}
            }],
            "input": [
                {
                    "type": "tool_search_call",
                    "id": "tsc_1",
                    "call_id": "call_search_1",
                    "arguments": {"query": "weather"}
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_search_1",
                    "tools": [{
                        "type": "mcp",
                        "server_label": "weather",
                        "server_description": "Weather tools",
                        "server_url": "https://mcp.example.test/mcp",
                        "defer_loading": true
                    }]
                }
            ]
        }))
        .expect("valid tool-search request");
        let ResponsesInput::Items(items) = &mut request.input else {
            panic!("test request uses item history")
        };
        let InputItem::ToolSearchOutput(output) = &mut items[1] else {
            panic!("test request contains a search output")
        };
        let ResponsesTool::Mcp(mcp) = &mut output.tools[0] else {
            panic!("test output returns MCP")
        };
        mcp.discovered_tools.push(McpDiscoveredToolParam {
            server_label: "weather".to_owned(),
            tool_name: "discovered-tool-sentinel".to_owned(),
            internal_name: "internal-name-sentinel".to_owned(),
            tool: serde_json::from_value(serde_json::json!({
                "name": "discovered-tool-sentinel",
                "description": "discovered-description-sentinel",
                "inputSchema": {
                    "type": "object",
                    "properties": {"discovered-schema-sentinel": {"type": "string"}}
                }
            }))
            .expect("valid discovered tool"),
        });

        let state = ToolSearchState::build(&request).expect("internal execution state remains valid");
        let private = state
            .private_inference_request(&request)
            .expect("active state materializes a private inference request");
        let private_input = serialize_to_string(&private.input).expect("private input serializes");
        for forbidden in [
            "_agentic_discovered_tools",
            "discovered-tool-sentinel",
            "internal-name-sentinel",
            "discovered-description-sentinel",
            "discovered-schema-sentinel",
        ] {
            assert!(!private_input.contains(forbidden), "private input leaked {forbidden}");
        }
    }
}
