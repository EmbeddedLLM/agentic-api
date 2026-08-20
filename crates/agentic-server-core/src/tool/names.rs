use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::types::tools::{CodexNamespaceMember, ResponsesTool};

use super::ToolError;

pub const MODEL_VISIBLE_NAMESPACE_MEMBER_PREFIX: &str = "agentic_ns__";
pub const MAX_MODEL_VISIBLE_TOOL_NAME_LEN: usize = 64;

const HASHED_NAMESPACE_MEMBER_SUFFIX_LEN: usize = 18;

fn stable_name_hash(value: &str) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    value.bytes().fold(FNV_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    })
}

#[must_use]
pub fn model_visible_namespace_member_name(namespace: &str, member: &str) -> String {
    let full_name = format!("{MODEL_VISIBLE_NAMESPACE_MEMBER_PREFIX}{namespace}__{member}");
    if full_name.chars().count() <= MAX_MODEL_VISIBLE_TOOL_NAME_LEN {
        return full_name;
    }

    let hash = stable_name_hash(&full_name);
    let readable_len = MAX_MODEL_VISIBLE_TOOL_NAME_LEN - HASHED_NAMESPACE_MEMBER_SUFFIX_LEN;
    let readable_prefix = full_name.chars().take(readable_len).collect::<String>();
    format!("{readable_prefix}__{hash:016x}")
}

enum DeclaredNameOrigin<'a> {
    TopLevel { description: &'static str },
    NamespaceMember { namespace: &'a str, member: &'a str },
}

impl DeclaredNameOrigin<'_> {
    fn description(&self) -> String {
        match self {
            Self::TopLevel { description } => (*description).to_owned(),
            Self::NamespaceMember { namespace, member } => {
                format!("Codex namespace member {namespace}.{member}")
            }
        }
    }
}

/// Validate the exact names that public declarations expose to the model.
///
/// This pass is intentionally declaration-only and performs no MCP discovery.
/// Discovered MCP member collisions remain a post-`tools/list` registry check.
///
/// # Errors
///
/// Returns [`ToolError::Config`] when function, custom, built-in, or normalized
/// namespace-member declarations resolve to the same model-visible name.
pub(crate) fn validate_model_visible_declared_names(tools: &[ResponsesTool]) -> Result<(), ToolError> {
    let mut names = HashMap::new();
    for tool in tools {
        match tool {
            ResponsesTool::Function(function) => {
                record_name(
                    &mut names,
                    function.name.as_str(),
                    DeclaredNameOrigin::TopLevel {
                        description: "function tool",
                    },
                )?;
            }
            ResponsesTool::Custom(custom) => {
                record_name(
                    &mut names,
                    custom.name.as_str(),
                    DeclaredNameOrigin::TopLevel {
                        description: "custom tool",
                    },
                )?;
            }
            ResponsesTool::WebSearch(_) => {
                record_name(
                    &mut names,
                    "web_search",
                    DeclaredNameOrigin::TopLevel {
                        description: "web search tool",
                    },
                )?;
            }
            ResponsesTool::FileSearch(_) => {
                record_name(
                    &mut names,
                    "file_search",
                    DeclaredNameOrigin::TopLevel {
                        description: "file search tool",
                    },
                )?;
            }
            ResponsesTool::CodeInterpreter(_) => {
                record_name(
                    &mut names,
                    "code_interpreter",
                    DeclaredNameOrigin::TopLevel {
                        description: "code interpreter tool",
                    },
                )?;
            }
            ResponsesTool::Namespace(namespace) => {
                for member in &namespace.tools {
                    let CodexNamespaceMember::Function(function) = member else {
                        continue;
                    };
                    let name = model_visible_namespace_member_name(&namespace.name, function.name.as_str());
                    record_name(
                        &mut names,
                        &name,
                        DeclaredNameOrigin::NamespaceMember {
                            namespace: &namespace.name,
                            member: function.name.as_str(),
                        },
                    )?;
                }
            }
            ResponsesTool::ToolSearch(_) | ResponsesTool::Mcp(_) | ResponsesTool::Unknown => {}
        }
    }
    Ok(())
}

fn record_name<'a>(
    names: &mut HashMap<String, DeclaredNameOrigin<'a>>,
    name: &str,
    origin: DeclaredNameOrigin<'a>,
) -> Result<(), ToolError> {
    match names.entry(name.to_owned()) {
        Entry::Vacant(entry) => {
            entry.insert(origin);
            Ok(())
        }
        Entry::Occupied(existing) => {
            let existing_description = existing.get().description();
            match origin {
                DeclaredNameOrigin::NamespaceMember { namespace, member } => Err(ToolError::Config(format!(
                    "codex namespace member {namespace}.{member} at generated name {name}, which collides with a declared {existing_description}"
                ))),
                DeclaredNameOrigin::TopLevel { description } => Err(ToolError::Config(format!(
                    "{description} model-visible name '{name}' collides with declared {existing_description}"
                ))),
            }
        }
    }
}
