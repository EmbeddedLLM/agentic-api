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
fn native_mcp_declaration_ignores_a_client_supplied_tool_name() {
    let tool = serde_json::from_value::<ResponsesTool>(serde_json::json!({
        "type": "mcp",
        "name": "increment",
        "server_label": "counter",
        "server_url": "http://127.0.0.1:8000/mcp"
    }))
    .expect("MCP declaration with an unknown field");

    let serialized = serde_json::to_value(tool).expect("serialized MCP declaration");
    assert_eq!(serialized["server_label"], "counter");
    assert!(serialized.get("name").is_none());
}
