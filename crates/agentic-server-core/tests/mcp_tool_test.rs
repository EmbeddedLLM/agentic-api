use agentic_core::tool::{GatewayExecutors, ToolRegistry, ToolType};
use agentic_core::types::tools::ResponsesTool;

fn native_mcp_declaration() -> ResponsesTool {
    serde_json::from_value(serde_json::json!({
        "type": "mcp",
        "server_label": "counter",
        "server_url": "http://127.0.0.1:8000/mcp",
        "allowed_tools": ["increment"],
        "require_approval": "never"
    }))
    .expect("native MCP declaration")
}

#[test]
fn native_mcp_declaration_uses_server_identity_without_a_tool_name() {
    let ResponsesTool::Mcp(param) = native_mcp_declaration() else {
        panic!("expected MCP declaration");
    };

    assert_eq!(param.server_label, "counter");
    assert_eq!(param.server_url.as_deref(), Some("http://127.0.0.1:8000/mcp"));
    assert_eq!(
        param.allowed_tools.as_deref(),
        Some(["increment".to_owned()].as_slice())
    );
    assert_eq!(param.require_approval.as_deref(), Some("never"));
}

#[test]
fn native_mcp_declaration_rejects_a_client_supplied_tool_name() {
    let result = serde_json::from_value::<ResponsesTool>(serde_json::json!({
        "type": "mcp",
        "name": "increment",
        "server_label": "counter",
        "server_url": "http://127.0.0.1:8000/mcp"
    }));

    assert!(result.is_err());
}

#[tokio::test]
async fn read_mcp_resource_function_is_client_owned() {
    let mut tools = vec![
        serde_json::from_value::<ResponsesTool>(serde_json::json!({
            "type": "function",
            "name": "read_mcp_resource",
            "description": "A client-owned function with no gateway MCP semantics",
            "parameters": {"type": "object"},
            "metadata": {
                "server_label": "repo",
                "server_url": "http://127.0.0.1:8000/mcp"
            }
        }))
        .expect("function declaration"),
    ];
    let mut executors = GatewayExecutors::default();

    let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
        .await
        .expect("function registry");
    let entry = registry.lookup("read_mcp_resource").expect("function registry entry");

    assert_eq!(entry.tool_type, ToolType::Function);
    assert!(entry.handler.is_none());
}
