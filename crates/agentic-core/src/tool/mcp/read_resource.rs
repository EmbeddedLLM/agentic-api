use serde::Deserialize;
use serde_json::json;

use crate::types::io::FunctionTool;

pub const READ_MCP_RESOURCE_TOOL_NAME: &str = "read_mcp_resource";

#[derive(Debug, Deserialize)]
pub struct ReadResourceArgs {
    pub server: String,
    pub uri: String,
}

#[must_use]
pub fn read_mcp_resource_spec() -> FunctionTool {
    FunctionTool {
        type_: "function".to_owned(),
        name: READ_MCP_RESOURCE_TOOL_NAME.to_owned(),
        description: Some("Read a resource by URI from a connected MCP server.".to_owned()),
        parameters: Some(json!({
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "The server_label to read from."
                },
                "uri": {
                    "type": "string",
                    "description": "The resource URI to read."
                }
            },
            "required": ["server", "uri"],
            "additionalProperties": false
        })),
        strict: Some(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_mcp_resource_spec_has_expected_shape() {
        let spec = read_mcp_resource_spec();

        assert_eq!(spec.type_, "function");
        assert_eq!(spec.name, READ_MCP_RESOURCE_TOOL_NAME);
        assert_eq!(
            spec.parameters.as_ref().unwrap()["required"],
            serde_json::json!(["server", "uri"])
        );
    }
}
