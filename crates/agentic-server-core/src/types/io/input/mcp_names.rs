//! The function names the model knows MCP tools by.
//!
//! Discovery names each server's tools in listing order, giving two tools whose names sanitize alike distinct names.
//! A listing record in history therefore says how the model knew a replayed call's tool, independently of the tools a
//! later request offers (a compaction summary offers none).

use std::collections::HashMap;

use super::InputItem;
use crate::types::io::output::McpListTools;
use crate::types::tools::ResponsesTool;

const INTERNAL_MCP_PREFIX: &str = "mcp__";
const MAX_INTERNAL_TOOL_NAME_LEN: usize = 64;

/// The function names of MCP tools, by server label and tool name.
#[derive(Clone, Default)]
pub(crate) struct McpToolNames {
    names: HashMap<(String, String), String>,
}

impl McpToolNames {
    /// The names discovery assigned this request, recorded on the declared servers.
    pub(crate) fn from_tools(tools: &[ResponsesTool]) -> Self {
        let names = tools
            .iter()
            .filter_map(|tool| match tool {
                ResponsesTool::Mcp(server) => Some(&server.discovered_tools),
                _ => None,
            })
            .flatten()
            .map(|tool| {
                (
                    (tool.server_label.clone(), tool.tool_name.clone()),
                    tool.internal_name.clone(),
                )
            })
            .collect();
        Self { names }
    }

    /// These names, completed by `other`'s for tools these do not name.
    #[must_use]
    pub(crate) fn or(mut self, other: &Self) -> Self {
        for (tool, name) in &other.names {
            self.names.entry(tool.clone()).or_insert_with(|| name.clone());
        }
        self
    }

    /// These names, completed by the names each server's latest successful listing in `items` gave its tools. This
    /// request's discovery names the tools the model can call now, so it wins; a listing names the rest, as when a
    /// request offers no tools (a compaction summary).
    pub(crate) fn with_listings<'i>(&self, items: impl IntoIterator<Item = &'i InputItem>) -> Self {
        let mut latest: HashMap<&str, &McpListTools> = HashMap::new();
        for item in items {
            if let InputItem::McpListTools(listing) = item
                && listing.error.is_none()
            {
                latest.insert(&listing.server_label, listing);
            }
        }
        let mut names = self.names.clone();
        for (server_label, listing) in latest {
            let mut used = HashMap::new();
            for tool in &listing.tools {
                let name = internal_mcp_tool_name(server_label, &tool.name, &mut used);
                names
                    .entry((server_label.to_owned(), tool.name.clone()))
                    .or_insert(name);
            }
        }
        Self { names }
    }

    pub(super) fn model_name(&self, server_label: &str, tool_name: &str) -> String {
        self.names
            .get(&(server_label.to_owned(), tool_name.to_owned()))
            .cloned()
            .unwrap_or_else(|| internal_mcp_tool_name(server_label, tool_name, &mut HashMap::new()))
    }
}

/// The function name discovery gives `tool_name` of `server_label`, unique among the names in `used`.
pub(crate) fn internal_mcp_tool_name(
    server_label: &str,
    tool_name: &str,
    used: &mut HashMap<String, (String, String)>,
) -> String {
    let identity = (server_label.to_owned(), tool_name.to_owned());
    let base = sanitize_internal_tool_name(&format!("{INTERNAL_MCP_PREFIX}{server_label}__{tool_name}"));
    if base.len() <= MAX_INTERNAL_TOOL_NAME_LEN && used.get(&base).is_none_or(|existing| existing == &identity) {
        used.insert(base.clone(), identity);
        return base;
    }

    let mut attempt = 0_u32;
    loop {
        let hash_input = if attempt == 0 {
            format!("{server_label}:{tool_name}")
        } else {
            format!("{server_label}:{tool_name}:{attempt}")
        };
        let suffix = format!("__{:010x}", stable_name_hash(&hash_input) & 0xff_ffff_ffff);
        let prefix_len = MAX_INTERNAL_TOOL_NAME_LEN.saturating_sub(suffix.len());
        let candidate = format!("{}{}", &base[..base.len().min(prefix_len)], suffix);
        if used.get(&candidate).is_none_or(|existing| existing == &identity) {
            used.insert(candidate.clone(), identity);
            return candidate;
        }
        attempt = attempt.saturating_add(1);
    }
}

fn sanitize_internal_tool_name(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

/// A stable 64-bit FNV-1a hash, for names that must not change across processes.
pub(crate) fn stable_name_hash(value: &str) -> u64 {
    value.as_bytes().iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::{Value, json};

    use super::{McpToolNames, internal_mcp_tool_name};
    use crate::types::io::InputItem;
    use crate::types::tools::ResponsesTool;

    fn listing(id: &str, tools: &[&str], error: Option<&str>) -> Value {
        let tools: Vec<_> = tools
            .iter()
            .map(|name| json!({"name": name, "input_schema": {"type": "object"}}))
            .collect();
        json!({"type": "mcp_list_tools", "id": id, "server_label": "docs", "tools": tools, "error": error})
    }

    fn items(value: Value) -> Vec<InputItem> {
        serde_json::from_value(value).expect("listing items")
    }

    fn collided_name() -> String {
        let mut used = HashMap::new();
        internal_mcp_tool_name("docs", "read.file", &mut used);
        internal_mcp_tool_name("docs", "read_file", &mut used)
    }

    #[test]
    fn the_latest_successful_listing_names_a_server_s_tools() {
        let history = items(json!([
            listing("mcpl_1", &["read_file"], None),
            listing("mcpl_2", &["read.file", "read_file"], None),
            listing("mcpl_3", &["read_file"], Some("unreachable"))
        ]));
        let names = McpToolNames::default().with_listings(&history);
        assert_eq!(names.model_name("docs", "read_file"), collided_name());
        assert_ne!(collided_name(), "mcp__docs__read_file");
    }

    /// History listed the server twice and this request rediscovered the first tool set, which the registry does not
    /// emit again: this request's names, which the model can call, win over the latest listing in history.
    #[test]
    fn this_request_s_discovery_wins_over_an_older_listing() {
        let history = items(json!([
            listing("mcpl_1", &["read_file"], None),
            listing("mcpl_2", &["read.file", "read_file"], None)
        ]));
        let names = discovered("read_file", "mcp__docs__read_file").with_listings(&history);
        assert_eq!(names.model_name("docs", "read_file"), "mcp__docs__read_file");
        assert_eq!(
            names.model_name("docs", "read.file"),
            "mcp__docs__read_file",
            "named by the listing"
        );
    }

    fn discovered(tool_name: &str, internal_name: &str) -> McpToolNames {
        let mut tools: Vec<ResponsesTool> = serde_json::from_value(json!([{
            "type": "mcp", "server_label": "docs", "server_url": "https://docs.example/mcp",
            "require_approval": "never"
        }]))
        .expect("declaration");
        let ResponsesTool::Mcp(server) = &mut tools[0] else {
            panic!("MCP declaration")
        };
        server.discovered_tools = vec![
            serde_json::from_value(
                json!({"server_label": "docs", "tool_name": tool_name, "internal_name": internal_name,
                "tool": {"name": tool_name, "inputSchema": {"type": "object"}}}),
            )
            .expect("discovered tool"),
        ];
        McpToolNames::from_tools(&tools)
    }

    #[test]
    fn without_a_listing_the_request_s_discovery_names_a_call_else_its_derived_name() {
        let names = discovered("read_file", "mcp__docs__discovered").with_listings(&[]);
        assert_eq!(names.model_name("docs", "read_file"), "mcp__docs__discovered");
        assert_eq!(names.model_name("docs", "search"), "mcp__docs__search");
    }
}
