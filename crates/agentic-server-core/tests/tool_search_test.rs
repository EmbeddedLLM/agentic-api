mod support;

use std::collections::HashSet;
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fs;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use agentic_core::executor::{ConversationHandler, ExecuteRequest, ExecutionContext, ResponseHandler};
use agentic_core::storage::{ConversationStore, ResponseStore, create_pool_with_schema};
use agentic_core::tool::{
    GatewayExecutor, ToolError, ToolHandler, ToolOutput, ToolType, model_visible_namespace_member_name,
};
use agentic_core::types::io::{FunctionTool, OutputItem};
use agentic_core::types::request_response::{RequestPayload, ResponsePayload};
use axum::Router;
use axum::body::Bytes;
use axum::response::IntoResponse;
use axum::routing::post;
use either::Either;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

#[derive(Default)]
struct McpRequestCounts {
    initialize: AtomicUsize,
    list_tools: AtomicUsize,
    call_tool: AtomicUsize,
}

#[derive(Debug)]
struct CountingWebSearch {
    calls: Arc<AtomicUsize>,
}

impl ToolHandler for CountingWebSearch {
    fn tool_type(&self) -> ToolType {
        ToolType::WebSearch
    }

    fn validate(&self, _param: &Value) -> Result<(), ToolError> {
        Ok(())
    }

    fn normalize(&self, _param: &Value) -> Vec<FunctionTool> {
        Vec::new()
    }
}

impl GatewayExecutor for CountingWebSearch {
    fn execute(
        &self,
        call_id: &str,
        _tool_name: &str,
        _arguments: &str,
        _config: &Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(std::future::ready(Ok(ToolOutput {
            call_id: call_id.to_owned(),
            output: "must not execute".to_owned(),
        })))
    }
}

async fn spawn_sequenced_llm(responses: Vec<Value>) -> (String, Arc<Mutex<Vec<Value>>>, tokio::task::JoinHandle<()>) {
    let responses = Arc::new(Mutex::new(VecDeque::from(responses)));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let route_responses = Arc::clone(&responses);
    let route_requests = Arc::clone(&requests);
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let route_responses = Arc::clone(&route_responses);
            let route_requests = Arc::clone(&route_requests);
            async move {
                route_requests
                    .lock()
                    .await
                    .push(serde_json::from_slice(&body).expect("captured request JSON"));
                let response = route_responses
                    .lock()
                    .await
                    .pop_front()
                    .expect("one prepared response per request");
                axum::response::Response::builder()
                    .status(200)
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(response.to_string()))
                    .expect("mock response")
                    .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock LLM");
    let address = listener.local_addr().expect("mock address");
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.expect("mock server") });
    (format!("http://{address}"), requests, handle)
}

async fn spawn_sequenced_streaming_llm(
    responses: Vec<String>,
) -> (String, Arc<Mutex<Vec<Value>>>, tokio::task::JoinHandle<()>) {
    let responses = Arc::new(Mutex::new(VecDeque::from(responses)));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let route_responses = Arc::clone(&responses);
    let route_requests = Arc::clone(&requests);
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let route_responses = Arc::clone(&route_responses);
            let route_requests = Arc::clone(&route_requests);
            async move {
                route_requests
                    .lock()
                    .await
                    .push(serde_json::from_slice(&body).expect("captured request JSON"));
                let response = route_responses
                    .lock()
                    .await
                    .pop_front()
                    .expect("one prepared response per request");
                axum::response::Response::builder()
                    .status(200)
                    .header("Content-Type", "text/event-stream")
                    .body(axum::body::Body::from(response))
                    .expect("mock response")
                    .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock LLM");
    let address = listener.local_addr().expect("mock address");
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.expect("mock server") });
    (format!("http://{address}"), requests, handle)
}

fn streaming_search_response(arguments: &str) -> String {
    let events = [
        json!({
            "type": "response.created", "sequence_number": 0,
            "response": {
                "id": "upstream_search", "status": "in_progress",
                "tools": [{
                    "type": "function", "name": "tool_search",
                    "description": "private normalized catalog",
                    "parameters": {"type": "object"}
                }]
            }
        }),
        json!({
            "type": "response.in_progress", "sequence_number": 1,
            "response": {
                "id": "upstream_search", "status": "in_progress",
                "tools": [{
                    "type": "function", "name": "tool_search",
                    "description": "private normalized catalog",
                    "parameters": {"type": "object"}
                }]
            }
        }),
        json!({
            "type": "response.output_item.added", "sequence_number": 2, "output_index": 0,
            "item": {"id": "fc_search_1", "type": "function_call", "call_id": "call_search_1",
                "name": "tool_search", "arguments": "", "status": "in_progress"}
        }),
        json!({
            "type": "response.function_call_arguments.delta", "sequence_number": 3, "output_index": 0,
            "item_id": "fc_search_1", "delta": arguments
        }),
        json!({
            "type": "response.function_call_arguments.done", "sequence_number": 4, "output_index": 0,
            "item_id": "fc_search_1", "name": "tool_search", "arguments": arguments
        }),
        json!({
            "type": "response.output_item.done", "sequence_number": 5, "output_index": 0,
            "item": {"id": "fc_search_1", "type": "function_call", "call_id": "call_search_1",
                "name": "tool_search", "arguments": arguments, "status": "completed"}
        }),
        json!({
            "type": "response.completed", "sequence_number": 6,
            "response": {
                "id": "upstream_search", "status": "completed", "usage": null,
                "output": [{
                    "type": "function_call", "id": "fc_provider_terminal_regenerated",
                    "call_id": "call_search_1", "name": "tool_search",
                    "arguments": arguments, "status": "completed"
                }]
            }
        }),
    ];
    streaming_response(events)
}

fn streaming_response(events: impl IntoIterator<Item = Value>) -> String {
    let mut response = String::new();
    for event in events {
        writeln!(&mut response, "data: {event}\n").expect("writing to String cannot fail");
    }
    response.push_str("data: [DONE]\n\n");
    response
}

fn streaming_partial_search_failure_response() -> String {
    let events = [
        json!({
            "type": "response.created",
            "response": {"id": "upstream_failed", "status": "in_progress"}
        }),
        json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"id": "fc_partial", "type": "function_call", "call_id": "call_partial",
                "arguments": "", "status": "in_progress"}
        }),
        json!({
            "type": "response.function_call_arguments.delta", "output_index": 0,
            "item_id": "fc_partial", "delta": "{\"query\":\"weather"
        }),
        json!({
            "type": "response.failed",
            "response": {
                "id": "upstream_failed", "status": "failed", "usage": null,
                "error": {"code": "provider_failure", "message": "provider stopped"},
                "incomplete_details": {"reason": "upstream_error"}
            }
        }),
    ];
    streaming_response(events)
}

fn streaming_pending_call_overflow_response() -> String {
    let mut events = vec![json!({
        "type": "response.created",
        "response": {"id": "upstream_pending_overflow", "status": "in_progress"}
    })];
    events.extend((0..=128).map(|output_index| {
        json!({
            "type": "response.output_item.added", "output_index": output_index,
            "item": {"id": format!("fc_pending_{output_index}"), "type": "function_call",
                "call_id": format!("call_pending_{output_index}"), "arguments": "", "status": "in_progress"}
        })
    }));
    streaming_response(events)
}

fn streaming_gateway_call_then_malformed_search_response() -> String {
    let events = [
        json!({"type":"response.created","response":{"id":"up_mixed","status":"in_progress"}}),
        json!({
            "type":"response.output_item.added","output_index":0,
            "item":{"id":"fc_web","type":"function_call","status":"in_progress",
                "name":"web_search","call_id":"call_web","arguments":""}
        }),
        json!({
            "type":"response.output_item.done","output_index":0,
            "item":{"id":"fc_web","type":"function_call","status":"completed",
                "name":"web_search","call_id":"call_web","arguments":"{\"query\":\"weather\"}"}
        }),
        json!({
            "type":"response.output_item.added","output_index":1,
            "item":{"id":"fc_search_bad","type":"function_call","status":"in_progress",
                "name":"tool_search","call_id":"call_search_bad","arguments":""}
        }),
        json!({
            "type":"response.function_call_arguments.delta","output_index":1,
            "item_id":"fc_search_bad","delta":"[]"
        }),
        json!({
            "type":"response.function_call_arguments.done","output_index":1,
            "item_id":"fc_search_bad","name":"tool_search","arguments":"[]"
        }),
    ];
    streaming_response(events)
}

fn streaming_named_function_response(name: &str) -> String {
    streaming_response([
        json!({
            "type": "response.created", "sequence_number": 0,
            "response": {"id": "upstream_withheld", "status": "in_progress"}
        }),
        json!({
            "type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
            "item": {"id": "fc_withheld", "type": "function_call", "call_id": "call_withheld",
                "name": name, "arguments": "", "status": "in_progress"}
        }),
    ])
}

fn streaming_late_named_function_response(name: &str) -> String {
    streaming_response([
        json!({
            "type": "response.created", "sequence_number": 0,
            "response": {"id": "upstream_withheld", "status": "in_progress"}
        }),
        json!({
            "type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
            "item": {"id": "fc_withheld", "type": "function_call", "call_id": "call_withheld",
                "arguments": "", "status": "in_progress"}
        }),
        json!({
            "type": "response.function_call_arguments.done", "sequence_number": 2, "output_index": 0,
            "item_id": "fc_withheld", "name": name, "arguments": "{}"
        }),
    ])
}

fn streaming_terminal_function_response(name: &str) -> String {
    streaming_response([
        json!({
            "type": "response.created", "sequence_number": 0,
            "response": {"id": "upstream_withheld", "status": "in_progress"}
        }),
        json!({
            "type": "response.completed", "sequence_number": 1,
            "response": {
                "id": "upstream_withheld", "status": "completed", "usage": null,
                "output": [{
                    "type": "function_call", "id": "fc_withheld", "call_id": "call_withheld",
                    "name": name, "arguments": "{}", "status": "completed"
                }]
            }
        }),
    ])
}

async fn run_streaming(request: RequestPayload, context: Arc<ExecutionContext>) -> Vec<Value> {
    let Either::Right(mut stream) = ExecuteRequest::new(request, context)
        .run()
        .await
        .expect("streaming execution")
    else {
        panic!("streaming request must return a stream")
    };
    let mut events = Vec::new();
    while let Some(chunk) = stream.next().await {
        for line in chunk.lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data != "[DONE]" {
                events.push(serde_json::from_str(data).expect("stream event JSON"));
            }
        }
    }
    events
}

async fn spawn_counting_mcp() -> (String, Arc<McpRequestCounts>, tokio::task::JoinHandle<()>) {
    let counts = Arc::new(McpRequestCounts::default());
    let route_counts = Arc::clone(&counts);
    let app = Router::new().route(
        "/mcp",
        post(move |body: Bytes| {
            let route_counts = Arc::clone(&route_counts);
            async move {
                let request: Value = serde_json::from_slice(&body).expect("MCP request JSON");
                let method = request["method"].as_str().unwrap_or_default();
                let id = request.get("id").cloned().unwrap_or(Value::Null);
                let response = match method {
                    "initialize" => {
                        route_counts.initialize.fetch_add(1, Ordering::SeqCst);
                        Some(json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "protocolVersion": "2025-06-18",
                                "capabilities": {"tools": {}},
                                "serverInfo": {"name": "counting-mcp", "version": "1.0.0"}
                            }
                        }))
                    }
                    "notifications/initialized" => None,
                    "tools/list" => {
                        route_counts.list_tools.fetch_add(1, Ordering::SeqCst);
                        Some(json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "tools": [{
                                    "name": "forecast",
                                    "description": "Return a forecast",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"city": {"type": "string"}}
                                    }
                                }]
                            }
                        }))
                    }
                    "tools/call" => {
                        route_counts.call_tool.fetch_add(1, Ordering::SeqCst);
                        Some(json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [{"type": "text", "text": "sunny"}],
                                "isError": false
                            }
                        }))
                    }
                    other => panic!("unexpected MCP method {other}"),
                };

                match response {
                    Some(response) => axum::response::Response::builder()
                        .status(200)
                        .header("Content-Type", "application/json")
                        .body(axum::body::Body::from(response.to_string()))
                        .expect("MCP response")
                        .into_response(),
                    None => axum::response::Response::builder()
                        .status(202)
                        .body(axum::body::Body::empty())
                        .expect("MCP notification response")
                        .into_response(),
                }
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind MCP server");
    let address = listener.local_addr().expect("MCP address");
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.expect("MCP server") });
    (format!("http://{address}/mcp"), counts, handle)
}

fn request(input: &Value, tools: &Value) -> RequestPayload {
    serde_json::from_value(json!({
        "model": "test",
        "input": input,
        "tools": tools,
        "store": false,
        "stream": false,
        "parallel_tool_calls": false
    }))
    .expect("public request")
}

async fn run(request: RequestPayload, context: Arc<ExecutionContext>) -> ResponsePayload {
    match Box::pin(ExecuteRequest::new(request, context).run())
        .await
        .expect("blocking execution")
    {
        Either::Left(response) => response,
        Either::Right(_) => panic!("blocking helper received a streaming response"),
    }
}

fn assert_public_search_call(response: &ResponsePayload) -> Value {
    let OutputItem::ToolSearchCall(search_call) = &response.output[0] else {
        panic!("normalized search call must be public")
    };
    assert_eq!(search_call.id, "tsc_search_1");
    assert_eq!(search_call.call_id, "call_search_1");
    assert_eq!(
        search_call.arguments,
        serde_json::from_value(json!({"query": "weather"})).unwrap()
    );
    let public_search_call = serde_json::to_value(&response.output[0]).expect("public search call serializes");
    assert_eq!(public_search_call["execution"], "client");
    assert_eq!(public_search_call["status"], "completed");
    public_search_call
}

fn assert_private_request_sequence(requests: &[Value]) {
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0]["tools"].as_array().map(Vec::len), Some(1));
    assert_eq!(requests[0]["tools"][0]["name"], "tool_search");
    for request in &requests[1..] {
        assert!(request.get("previous_response_id").is_none());
        assert_eq!(request["input"][1]["type"], "function_call");
        assert_eq!(request["input"][1]["name"], "tool_search");
        assert_eq!(request["input"][1]["call_id"], "call_search_1");
        assert_eq!(request["input"][2]["type"], "function_call_output");
        assert_eq!(request["input"][2]["call_id"], "call_search_1");
        assert_eq!(request["tools"].as_array().map(Vec::len), Some(1));
        assert_eq!(request["tools"][0]["name"], "get_weather");
        assert!(request["tools"][0].get("defer_loading").is_none());
    }
}

fn assert_mcp_private_request_sequence(requests: &[Value], mcp_url: &str) {
    assert_eq!(requests.len(), 3);
    let initial_tool_names = requests[0]["tools"]
        .as_array()
        .expect("initial private tools")
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(initial_tool_names, ["tool_search"]);
    assert!(!requests[0].to_string().contains(mcp_url));
    for request in &requests[1..] {
        let tools = request["tools"].as_array().expect("private tools");
        let tool_names = tools
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect::<Vec<_>>();
        assert_eq!(tool_names, ["tool_search", "mcp__weather__forecast"]);
        assert!(tools.iter().all(|tool| tool.get("defer_loading").is_none()));
    }
}

fn search_declaration() -> Value {
    json!({
        "type": "tool_search",
        "execution": "client",
        "description": "Search the client tool catalog",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    })
}

fn deferred_weather() -> Value {
    json!({
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        },
        "strict": true,
        "defer_loading": true
    })
}

fn deferred_weather_mcp(server_url: &str) -> Value {
    json!({
        "type": "mcp",
        "server_label": "weather",
        "server_description": "Weather server",
        "server_url": server_url,
        "allowed_tools": ["forecast"],
        "require_approval": "never",
        "defer_loading": true
    })
}

fn completed_search_replay() -> Value {
    json!([
        {
            "type": "tool_search_call",
            "id": "tsc_prior",
            "call_id": "call_prior",
            "execution": "client",
            "arguments": {"query": "weather"},
            "status": "completed"
        },
        {
            "type": "tool_search_output",
            "call_id": "call_prior",
            "execution": "client",
            "status": "completed",
            "tools": [deferred_weather()]
        }
    ])
}

fn weather_namespace() -> Value {
    json!({
        "type": "namespace",
        "name": "weather_namespace_with_a_name_long_enough_to_require_bounded_flattening",
        "description": "Weather tools",
        "tools": [
            {
                "type": "function",
                "name": "current",
                "parameters": {"type": "object"}
            },
            {
                "type": "function",
                "name": "forecast_member_with_a_name_long_enough_to_require_bounded_flattening",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                },
                "defer_loading": true
            }
        ]
    })
}

fn loaded_weather_namespace_subset() -> Value {
    let namespace = weather_namespace();
    json!({
        "type": "namespace",
        "name": namespace["name"],
        "description": namespace["description"],
        "tools": [namespace["tools"][1].clone()]
    })
}

#[tokio::test]
async fn function_only_streaming_search_has_public_lifecycle_terminal_and_private_lowering() {
    let (llm_url, requests, _server) =
        spawn_sequenced_streaming_llm(vec![streaming_search_response(r#"{"query":"weather"}"#)]).await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));
    let mut payload = request(
        &json!("find weather"),
        &json!([search_declaration(), deferred_weather()]),
    );
    payload.stream = true;

    let events = run_streaming(payload, context).await;

    let response_envelopes = events
        .iter()
        .filter_map(|event| event.get("response"))
        .filter(|response| response.get("tools").is_some())
        .collect::<Vec<_>>();
    assert!(!response_envelopes.is_empty());
    for response in response_envelopes {
        assert_eq!(
            response["tools"],
            json!([search_declaration(), deferred_weather()]),
            "stream response envelopes must restore public tool declarations"
        );
        assert!(!response["tools"].to_string().contains("\"name\":\"tool_search\""));
    }

    assert_eq!(
        events
            .iter()
            .map(|event| event["sequence_number"].as_u64())
            .collect::<Vec<_>>(),
        (0..u64::try_from(events.len()).unwrap()).map(Some).collect::<Vec<_>>()
    );
    assert!(events.iter().all(|event| {
        !matches!(
            event["type"].as_str(),
            Some("response.function_call_arguments.delta" | "response.function_call_arguments.done" | "error")
        )
    }));
    let lifecycle = events
        .iter()
        .filter(|event| {
            matches!(
                event["type"].as_str(),
                Some("response.output_item.added" | "response.output_item.done")
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(lifecycle.len(), 2);
    assert_eq!(lifecycle[0]["item"]["type"], "tool_search_call");
    assert_eq!(lifecycle[0]["item"]["status"], "in_progress");
    assert_eq!(lifecycle[0]["item"]["arguments"], json!({}));
    assert_eq!(lifecycle[1]["item"]["type"], "tool_search_call");
    assert_eq!(lifecycle[1]["item"]["status"], "completed");
    assert_eq!(lifecycle[1]["item"]["arguments"], json!({"query": "weather"}));
    assert_eq!(lifecycle[0]["item"]["id"], lifecycle[1]["item"]["id"]);
    assert_eq!(lifecycle[0]["item"]["call_id"], lifecycle[1]["item"]["call_id"]);

    let terminal = events
        .iter()
        .find(|event| event["type"] == "response.completed")
        .expect("terminal response");
    let terminal_call = &terminal["response"]["output"][0];
    assert_eq!(terminal_call, &lifecycle[1]["item"]);

    let captured = requests.lock().await;
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0]["stream"], true);
    assert_eq!(captured[0]["tools"].as_array().map(Vec::len), Some(1));
    assert_eq!(captured[0]["tools"][0]["name"], "tool_search");
    assert!(captured[0].to_string().contains("get_weather"));
    assert!(!captured[0].to_string().contains("\"city\""));
}

#[tokio::test]
async fn malformed_streaming_search_finishes_with_response_failed_not_completed() {
    let (llm_url, _requests, _server) = spawn_sequenced_streaming_llm(vec![streaming_search_response("[]")]).await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));
    let mut payload = request(
        &json!("find weather"),
        &json!([search_declaration(), deferred_weather()]),
    );
    payload.stream = true;

    let events = run_streaming(payload, context).await;

    assert!(events.iter().all(|event| event["type"] != "response.completed"));
    assert!(events.iter().all(|event| {
        !matches!(
            event["type"].as_str(),
            Some("response.function_call_arguments.delta" | "response.function_call_arguments.done" | "error")
        )
    }));
    let failed = events.last().expect("response.failed");
    assert_eq!(failed["type"], "response.failed");
    assert_eq!(failed["response"]["status"], "failed");
    assert_eq!(failed["response"]["error"]["type"], "tool_error");
    assert_eq!(failed["response"]["error"]["code"], "tool_error");
    assert_eq!(
        events
            .iter()
            .map(|event| event["sequence_number"].as_u64())
            .collect::<Vec<_>>(),
        (0..u64::try_from(events.len()).unwrap()).map(Some).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn withheld_function_streams_fail_without_public_call_or_persistence() {
    let namespace = weather_namespace();
    let flat_name = model_visible_namespace_member_name(
        namespace["name"].as_str().expect("namespace name"),
        namespace["tools"][1]["name"].as_str().expect("member name"),
    );
    let cases = [
        (
            "get_weather".to_owned(),
            json!([search_declaration(), deferred_weather()]),
        ),
        (flat_name, json!([search_declaration(), namespace])),
    ];

    for (name, tools) in cases {
        for response in [
            streaming_named_function_response(&name),
            streaming_late_named_function_response(&name),
            streaming_terminal_function_response(&name),
        ] {
            let (llm_url, requests, _server) = spawn_sequenced_streaming_llm(vec![response]).await;
            let pool = create_pool_with_schema(Some("sqlite://?mode=memory"))
                .await
                .expect("storage schema");
            let context = Arc::new(ExecutionContext::new(
                ConversationHandler::new(ConversationStore::new(Arc::clone(&pool))),
                ResponseHandler::new(ResponseStore::new(Arc::clone(&pool))),
                Arc::new(reqwest::Client::new()),
                llm_url,
            ));
            let mut payload = request(&json!("find a tool"), &tools);
            payload.stream = true;
            payload.store = true;

            let events = run_streaming(payload, context).await;

            let failed = events.last().expect("response.failed");
            assert_eq!(failed["type"], "response.failed", "{name}");
            assert_eq!(failed["response"]["error"]["code"], "tool_error", "{name}");
            assert!(events.iter().all(|event| event["type"] != "response.completed"));
            assert!(events.iter().all(|event| {
                !matches!(
                    event["type"].as_str(),
                    Some("response.function_call_arguments.delta" | "response.function_call_arguments.done")
                ) && event
                    .get("item")
                    .and_then(|item| item.get("name"))
                    .and_then(Value::as_str)
                    != Some(name.as_str())
            }));
            let response_id = failed["response"]["id"].as_str().expect("failed response ID");
            let error = ResponseStore::new(pool)
                .get(response_id)
                .await
                .expect_err("invalid streamed response must not persist");
            assert!(error.is_not_found());
            assert_eq!(requests.lock().await.len(), 1, "only inference may run");
        }
    }
}

#[tokio::test]
async fn upstream_failure_after_partial_search_preserves_provider_error_without_normalized_output() {
    let (llm_url, _requests, _server) =
        spawn_sequenced_streaming_llm(vec![streaming_partial_search_failure_response()]).await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));
    let mut immediate_weather = deferred_weather();
    immediate_weather
        .as_object_mut()
        .expect("function tool")
        .remove("defer_loading");
    let mut payload = request(
        &json!("find weather"),
        &json!([search_declaration(), immediate_weather]),
    );
    payload.stream = true;

    let events = run_streaming(payload, context).await;

    let failed = events.last().expect("response.failed");
    assert_eq!(failed["type"], "response.failed");
    assert_eq!(failed["response"]["error"]["code"], "provider_failure");
    assert_eq!(failed["response"]["error"]["message"], "provider stopped");
    assert_eq!(failed["response"]["incomplete_details"]["reason"], "upstream_error");
    assert_eq!(failed["response"]["output"], json!([]));
    assert!(events.iter().all(|event| event["type"] != "response.completed"));
    assert!(events.iter().all(|event| {
        !matches!(
            event["type"].as_str(),
            Some("response.function_call_arguments.delta" | "response.function_call_arguments.done")
        )
    }));
}

#[tokio::test]
async fn search_active_pending_call_overflow_uses_standard_response_failed() {
    let (llm_url, _requests, _server) =
        spawn_sequenced_streaming_llm(vec![streaming_pending_call_overflow_response()]).await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));
    let mut payload = request(
        &json!("find weather"),
        &json!([search_declaration(), deferred_weather()]),
    );
    payload.stream = true;

    let events = run_streaming(payload, context).await;

    let failed = events.last().expect("response.failed");
    assert_eq!(failed["type"], "response.failed");
    assert_eq!(failed["response"]["error"]["code"], "tool_error");
    assert!(events.iter().all(|event| event["type"] != "error"));
    assert!(events.iter().all(|event| event["type"] != "response.completed"));
    assert_eq!(
        events
            .iter()
            .map(|event| event["sequence_number"].as_u64())
            .collect::<Vec<_>>(),
        (0..u64::try_from(events.len()).unwrap()).map(Some).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn malformed_streaming_search_is_not_dispatched_or_persisted_after_start() {
    let (llm_url, _requests, _server) =
        spawn_sequenced_streaming_llm(vec![streaming_gateway_call_then_malformed_search_response()]).await;
    let pool = create_pool_with_schema(Some("sqlite://?mode=memory"))
        .await
        .expect("storage schema");
    let calls = Arc::new(AtomicUsize::new(0));
    let context = Arc::new(
        ExecutionContext::new(
            ConversationHandler::new(ConversationStore::new(Arc::clone(&pool))),
            ResponseHandler::new(ResponseStore::new(Arc::clone(&pool))),
            Arc::new(reqwest::Client::new()),
            llm_url,
        )
        .with_gateway_executor(Arc::new(CountingWebSearch {
            calls: Arc::clone(&calls),
        })),
    );
    let mut payload = request(
        &json!("find weather"),
        &json!([search_declaration(), {"type":"web_search_preview"}]),
    );
    payload.stream = true;
    payload.store = true;

    let events = run_streaming(payload, context).await;

    let failed = events.last().expect("response.failed");
    assert_eq!(failed["type"], "response.failed");
    assert!(events.iter().all(|event| event["type"] != "response.completed"));
    assert_eq!(calls.load(Ordering::SeqCst), 0, "gateway call must not dispatch");
    let response_id = failed["response"]["id"].as_str().expect("failed response ID");
    let error = ResponseStore::new(pool)
        .get(response_id)
        .await
        .expect_err("malformed streamed response must not persist");
    assert!(error.is_not_found());
}

#[tokio::test]
async fn function_only_nonstreaming_manual_three_request_flow() {
    let (llm_url, requests, _server) = spawn_sequenced_llm(vec![
        json!({
            "id": "upstream_search",
            "object": "response",
            "status": "completed",
            "model": "test",
            "created_at": 0,
            "output": [{
                "type": "function_call",
                "id": "fc_search_1",
                "call_id": "call_search_1",
                "name": "tool_search",
                "arguments": "{\"query\":\"weather\"}",
                "status": "completed"
            }]
        }),
        json!({
            "id": "upstream_weather",
            "object": "response",
            "status": "completed",
            "model": "test",
            "created_at": 0,
            "output": [{
                "type": "function_call",
                "id": "fc_weather_1",
                "call_id": "call_weather_1",
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}",
                "status": "completed"
            }]
        }),
        json!({
            "id": "upstream_final",
            "object": "response",
            "status": "completed",
            "model": "test",
            "created_at": 0,
            "output": [{
                "type": "message",
                "id": "msg_final",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "PARIS_WEATHER_OK", "annotations": []}]
            }]
        }),
    ])
    .await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));
    let user = json!({"type": "message", "role": "user", "content": "find weather"});

    let first = run(
        request(
            &json!([user.clone()]),
            &json!([search_declaration(), deferred_weather()]),
        ),
        Arc::clone(&context),
    )
    .await;
    let public_search_call = assert_public_search_call(&first);

    let public_search_output = json!({
        "type": "tool_search_output",
        "call_id": "call_search_1",
        "tools": [deferred_weather()]
    });
    let second_input = json!([user.clone(), public_search_call.clone(), public_search_output.clone()]);
    let second = run(request(&second_input, &json!([])), Arc::clone(&context)).await;
    let OutputItem::FunctionCall(weather_call) = &second.output[0] else {
        panic!("loaded client function remains an ordinary public function call")
    };
    assert_eq!(weather_call.name, "get_weather");
    assert_eq!(weather_call.call_id, "call_weather_1");

    let public_weather_call = serde_json::to_value(&second.output[0]).expect("function call serializes");
    let mut third_items = second_input.as_array().expect("item history").clone();
    third_items.push(public_weather_call);
    third_items.push(json!({
        "type": "function_call_output",
        "call_id": "call_weather_1",
        "output": "sunny"
    }));
    let third = run(request(&Value::Array(third_items), &json!([])), context).await;
    assert!(matches!(&third.output[0], OutputItem::Message(_)));
    assert_eq!(
        serde_json::to_value(&third.output[0]).unwrap()["content"][0]["text"],
        "PARIS_WEATHER_OK"
    );

    assert_private_request_sequence(&requests.lock().await);
}

#[tokio::test]
async fn deferred_mcp_load_uses_existing_discovery_lifecycle_and_dispatch_once() {
    let (mcp_url, mcp_counts, _mcp_server) = spawn_counting_mcp().await;
    let (llm_url, requests, _llm_server) = spawn_sequenced_llm(vec![
        json!({
            "id": "upstream_search",
            "object": "response",
            "status": "completed",
            "model": "test",
            "created_at": 0,
            "output": [{
                "type": "function_call",
                "id": "fc_search_weather",
                "call_id": "call_search_weather",
                "name": "tool_search",
                "arguments": "{\"query\":\"weather\"}",
                "status": "completed"
            }]
        }),
        json!({
            "id": "upstream_mcp_call",
            "object": "response",
            "status": "completed",
            "model": "test",
            "created_at": 0,
            "output": [{
                "type": "function_call",
                "id": "fc_forecast",
                "call_id": "call_forecast",
                "name": "mcp__weather__forecast",
                "arguments": "{\"city\":\"Paris\"}",
                "status": "completed"
            }]
        }),
        json!({
            "id": "upstream_final",
            "object": "response",
            "status": "completed",
            "model": "test",
            "created_at": 0,
            "output": [{
                "type": "message",
                "id": "msg_final",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "MCP_WEATHER_OK", "annotations": []}]
            }]
        }),
    ])
    .await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));
    let deferred_mcp = deferred_weather_mcp(&mcp_url);
    let user = json!({"type": "message", "role": "user", "content": "find weather"});
    let tools = json!([search_declaration(), deferred_mcp.clone()]);

    let search_response = run(request(&json!([user.clone()]), &tools), Arc::clone(&context)).await;
    assert_eq!(mcp_counts.initialize.load(Ordering::SeqCst), 0);
    assert_eq!(mcp_counts.list_tools.load(Ordering::SeqCst), 0);
    assert_eq!(mcp_counts.call_tool.load(Ordering::SeqCst), 0);
    let public_search_call = serde_json::to_value(&search_response.output[0]).expect("search call serializes");
    assert_eq!(public_search_call["type"], "tool_search_call");
    assert_eq!(public_search_call["call_id"], "call_search_weather");

    let payload = request(
        &json!([
            user,
            public_search_call,
            {
                "type": "tool_search_output",
                "call_id": "call_search_weather",
                "tools": [deferred_mcp.clone()]
            }
        ]),
        &tools,
    );

    let response = run(payload, context).await;

    assert_eq!(mcp_counts.initialize.load(Ordering::SeqCst), 1);
    assert_eq!(mcp_counts.list_tools.load(Ordering::SeqCst), 1);
    assert_eq!(mcp_counts.call_tool.load(Ordering::SeqCst), 1);
    assert!(matches!(&response.output[0], OutputItem::McpListTools(_)));
    let OutputItem::McpCall(call) = &response.output[1] else {
        panic!("loaded MCP call must use the normal public MCP lifecycle")
    };
    assert_eq!(call.server_label, "weather");
    assert_eq!(call.name, "forecast");
    assert_eq!(call.output.as_deref(), Some("sunny"));
    assert!(matches!(&response.output[2], OutputItem::Message(_)));

    assert_mcp_private_request_sequence(&requests.lock().await, &mcp_url);
}

#[tokio::test]
async fn deferred_and_dynamic_mcp_discovery_rejects_matching_history_call_before_load() {
    for declared_before_search in [true, false] {
        let (mcp_url, mcp_counts, _mcp_server) = spawn_counting_mcp().await;
        let (llm_url, requests, _llm_server) = spawn_sequenced_llm(Vec::new()).await;
        let context = Arc::new(ExecutionContext::new(
            ConversationHandler::new(ConversationStore::disabled()),
            ResponseHandler::new(ResponseStore::disabled()),
            Arc::new(reqwest::Client::new()),
            llm_url,
        ));
        let deferred_mcp = deferred_weather_mcp(&mcp_url);
        let tools = if declared_before_search {
            json!([search_declaration(), deferred_mcp.clone()])
        } else {
            json!([search_declaration()])
        };
        let payload = request(
            &json!([
                {
                    "type": "function_call", "id": "fc_early", "call_id": "call_early",
                    "name": "mcp__weather__forecast", "arguments": "{}", "status": "completed"
                },
                {"type": "function_call_output", "call_id": "call_early", "output": "not executed"},
                {
                    "type": "tool_search_call", "id": "tsc_load", "call_id": "call_load",
                    "arguments": {"query": "weather"}
                },
                {"type": "tool_search_output", "call_id": "call_load", "tools": [deferred_mcp]}
            ]),
            &tools,
        );

        let Err(error) = Box::pin(ExecuteRequest::new(payload, context).run()).await else {
            panic!("an MCP function cannot be called before its server is loaded")
        };

        assert_eq!(error.http_status(), http::StatusCode::BAD_REQUEST);
        assert_eq!(mcp_counts.initialize.load(Ordering::SeqCst), 1);
        assert_eq!(mcp_counts.list_tools.load(Ordering::SeqCst), 1);
        assert_eq!(mcp_counts.call_tool.load(Ordering::SeqCst), 0);
        assert!(requests.lock().await.is_empty(), "inference must not run");
    }
}

#[tokio::test]
async fn immediate_mcp_stays_available_before_an_identical_search_result() {
    let (mcp_url, mcp_counts, _mcp_server) = spawn_counting_mcp().await;
    let (llm_url, requests, _llm_server) = spawn_sequenced_llm(vec![json!({
        "id": "upstream_final", "object": "response", "status": "completed", "model": "test",
        "created_at": 0, "output": [{
            "type": "message", "id": "msg_final", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "IMMEDIATE_MCP_OK", "annotations": []}]
        }]
    })])
    .await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));
    let mut immediate_mcp = deferred_weather_mcp(&mcp_url);
    immediate_mcp
        .as_object_mut()
        .expect("MCP declaration")
        .remove("defer_loading");
    let payload = request(
        &json!([
            {
                "type": "function_call", "id": "fc_early", "call_id": "call_early",
                "name": "mcp__weather__forecast", "arguments": "{}", "status": "completed"
            },
            {"type": "function_call_output", "call_id": "call_early", "output": "already handled"},
            {
                "type": "tool_search_call", "id": "tsc_identical", "call_id": "call_identical",
                "arguments": {"query": "weather"}
            },
            {
                "type": "tool_search_output", "call_id": "call_identical",
                "tools": [immediate_mcp.clone()]
            }
        ]),
        &json!([search_declaration(), immediate_mcp]),
    );

    let response = run(payload, context).await;

    assert!(matches!(&response.output[1], OutputItem::Message(_)));
    assert_eq!(mcp_counts.initialize.load(Ordering::SeqCst), 1);
    assert_eq!(mcp_counts.list_tools.load(Ordering::SeqCst), 1);
    assert_eq!(mcp_counts.call_tool.load(Ordering::SeqCst), 0);
    assert_eq!(requests.lock().await.len(), 1, "request must reach inference");
}

#[tokio::test]
async fn loaded_deferred_mcp_failure_uses_sanitized_public_list_tools_item() {
    let (llm_url, requests, _llm_server) = spawn_sequenced_llm(vec![json!({
        "id": "upstream_after_mcp_failure",
        "object": "response",
        "status": "completed",
        "model": "test",
        "created_at": 0,
        "output": [{
            "type": "message",
            "id": "msg_final",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "MCP_FAILURE_HANDLED", "annotations": []}]
        }]
    })])
    .await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));
    let deferred_mcp = json!({
        "type": "mcp",
        "server_label": "private_weather",
        "server_description": "Private weather server",
        "server_url": "http://url-user:url-password@127.0.0.1:1/mcp?token=query-secret",
        "headers": {"X-Private-Token": "header-secret"},
        "authorization": "authorization-secret",
        "require_approval": "never",
        "defer_loading": true
    });
    let payload = request(
        &json!([
            {
                "type": "tool_search_call",
                "id": "tsc_private_weather",
                "call_id": "call_search_private_weather",
                "arguments": {"query": "weather"}
            },
            {
                "type": "tool_search_output",
                "call_id": "call_search_private_weather",
                "tools": [deferred_mcp.clone()]
            }
        ]),
        &json!([search_declaration(), deferred_mcp]),
    );

    let response = run(payload, context).await;

    let OutputItem::McpListTools(list_tools) = &response.output[0] else {
        panic!("loaded MCP configuration failure must retain public list-tools semantics")
    };
    assert_eq!(list_tools.server_label, "private_weather");
    assert!(list_tools.tools.is_empty());
    assert_eq!(
        list_tools.error.as_deref(),
        Some("MCP server 'private_weather' failed to connect or list tools")
    );
    assert!(matches!(&response.output[1], OutputItem::Message(_)));

    let public_response = serde_json::to_string(&response).expect("response serializes");
    let upstream_requests = requests.lock().await;
    assert_eq!(upstream_requests.len(), 1);
    let upstream_request = upstream_requests[0].to_string();
    for secret in [
        "url-user",
        "url-password",
        "query-secret",
        "header-secret",
        "authorization-secret",
    ] {
        assert!(!public_response.contains(secret), "public response leaked {secret}");
        assert!(!upstream_request.contains(secret), "upstream request leaked {secret}");
    }
}

#[tokio::test]
async fn loaded_mcp_cannot_collide_with_a_still_withheld_function() {
    let (mcp_url, mcp_counts, _mcp_server) = spawn_counting_mcp().await;
    let (llm_url, requests, _llm_server) = spawn_sequenced_llm(Vec::new()).await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));
    let deferred_mcp = deferred_weather_mcp(&mcp_url);
    let payload = request(
        &json!([
            {
                "type": "tool_search_call",
                "id": "tsc_collision",
                "call_id": "call_search_collision",
                "arguments": {"query": "weather"}
            },
            {
                "type": "tool_search_output",
                "call_id": "call_search_collision",
                "tools": [deferred_mcp.clone()]
            }
        ]),
        &json!([
            search_declaration(),
            {
                "type": "function",
                "name": "mcp__weather__forecast",
                "parameters": {"type": "object"},
                "defer_loading": true
            },
            deferred_mcp
        ]),
    );

    let Err(error) = Box::pin(ExecuteRequest::new(payload, context).run()).await else {
        panic!("loaded MCP discovered name must not overwrite a declared function")
    };

    assert_eq!(error.http_status(), http::StatusCode::BAD_REQUEST);
    assert_eq!(mcp_counts.initialize.load(Ordering::SeqCst), 1);
    assert_eq!(mcp_counts.list_tools.load(Ordering::SeqCst), 1);
    assert_eq!(mcp_counts.call_tool.load(Ordering::SeqCst), 0);
    assert!(requests.lock().await.is_empty(), "collision must fail before inference");
}

#[tokio::test]
async fn function_only_nonstreaming_malformed_search_is_atomic_before_gateway_side_effects() {
    for (case, malformed_search) in [
        (
            "non-object arguments",
            json!({"arguments": "null", "namespace": null, "status": "completed"}),
        ),
        (
            "nonterminal status",
            json!({"arguments": "{}", "namespace": null, "status": "in_progress"}),
        ),
        (
            "unexpected namespace",
            json!({"arguments": "{}", "namespace": "tools", "status": "completed"}),
        ),
    ] {
        let (llm_url, requests, _server) = spawn_sequenced_llm(vec![json!({
            "id": "upstream_invalid_mixed",
            "object": "response",
            "status": "completed",
            "model": "test",
            "created_at": 0,
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_web",
                    "call_id": "call_web",
                    "name": "web_search",
                    "arguments": "{\"query\":\"weather\"}",
                    "status": "completed"
                },
                {
                    "type": "function_call",
                    "id": "fc_search",
                    "call_id": "call_search",
                    "name": "tool_search",
                    "arguments": malformed_search["arguments"],
                    "namespace": malformed_search["namespace"],
                    "status": malformed_search["status"]
                }
            ]
        })])
        .await;
        let calls = Arc::new(AtomicUsize::new(0));
        let context = Arc::new(
            ExecutionContext::new(
                ConversationHandler::new(ConversationStore::disabled()),
                ResponseHandler::new(ResponseStore::disabled()),
                Arc::new(reqwest::Client::new()),
                llm_url,
            )
            .with_gateway_executor(Arc::new(CountingWebSearch {
                calls: Arc::clone(&calls),
            })),
        );
        let request = request(
            &json!("find weather"),
            &json!([search_declaration(), {"type": "web_search_preview"}]),
        );

        let Err(error) = ExecuteRequest::new(request, context).run().await else {
            panic!("{case}: malformed reserved call must reject the whole response")
        };

        assert_eq!(error.http_status(), http::StatusCode::BAD_GATEWAY, "{case}");
        assert_eq!(error.error_type(), "tool_error", "{case}");
        assert_eq!(calls.load(Ordering::SeqCst), 0, "{case}: gateway call must not execute");
        assert_eq!(requests.lock().await.len(), 1, "{case}: only inference may run");
    }
}

#[tokio::test]
async fn declaration_free_replay_malformed_search_is_atomic_before_gateway_side_effects() {
    let (llm_url, requests, _server) = spawn_sequenced_llm(vec![json!({
        "id": "upstream_invalid_replay",
        "object": "response",
        "status": "completed",
        "model": "test",
        "created_at": 0,
        "output": [
            {
                "type": "function_call", "id": "fc_web", "call_id": "call_web",
                "name": "web_search", "arguments": "{\"query\":\"weather\"}", "status": "completed"
            },
            {
                "type": "function_call", "id": "fc_search", "call_id": "call_search",
                "name": "tool_search", "namespace": null, "arguments": "{}", "status": "in_progress"
            }
        ]
    })])
    .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let context = Arc::new(
        ExecutionContext::new(
            ConversationHandler::new(ConversationStore::disabled()),
            ResponseHandler::new(ResponseStore::disabled()),
            Arc::new(reqwest::Client::new()),
            llm_url,
        )
        .with_gateway_executor(Arc::new(CountingWebSearch {
            calls: Arc::clone(&calls),
        })),
    );
    let request = request(&completed_search_replay(), &json!([{"type": "web_search_preview"}]));

    let Err(error) = Box::pin(ExecuteRequest::new(request, context).run()).await else {
        panic!("malformed declaration-free replay call must reject the whole response")
    };
    assert_eq!(error.http_status(), http::StatusCode::BAD_GATEWAY);
    assert_eq!(calls.load(Ordering::SeqCst), 0, "gateway call must not execute");
    assert_eq!(requests.lock().await.len(), 1, "only inference may run");
}

#[tokio::test]
async fn namespace_nonstreaming_manual_flow_reuses_flattening_and_restoration() {
    let namespace = weather_namespace();
    let namespace_name = namespace["name"].as_str().expect("namespace name");
    let member_name = namespace["tools"][1]["name"].as_str().expect("member name");
    let flat_name = model_visible_namespace_member_name(namespace_name, member_name);
    assert!(flat_name.len() <= 64);
    let (llm_url, requests, _server) = spawn_sequenced_llm(vec![
        json!({
            "id": "upstream_search", "object": "response", "status": "completed", "model": "test",
            "created_at": 0, "output": [{
                "type": "function_call", "id": "fc_search", "call_id": "call_search",
                "name": "tool_search", "namespace": null, "arguments": "{\"query\":\"forecast\"}",
                "status": "completed"
            }]
        }),
        json!({
            "id": "upstream_namespace", "object": "response", "status": "completed", "model": "test",
            "created_at": 0, "output": [{
                "type": "function_call", "id": "fc_forecast", "call_id": "call_forecast",
                "name": flat_name, "namespace": null, "arguments": "{\"city\":\"Paris\"}",
                "status": "completed"
            }]
        }),
        json!({
            "id": "upstream_final", "object": "response", "status": "completed", "model": "test",
            "created_at": 0, "output": [{
                "type": "message", "id": "msg_final", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "NAMESPACE_OK", "annotations": []}]
            }]
        }),
    ])
    .await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));
    let user = json!({"type": "message", "role": "user", "content": "find forecast"});
    let first = run(
        request(
            &json!([user.clone()]),
            &json!([search_declaration(), namespace.clone()]),
        ),
        Arc::clone(&context),
    )
    .await;
    let public_search_call = serde_json::to_value(&first.output[0]).expect("search call serializes");
    let public_search_output = json!({
        "type": "tool_search_output", "call_id": "call_search", "execution": "client",
        "status": "completed", "tools": [loaded_weather_namespace_subset()]
    });
    let second_input = json!([user.clone(), public_search_call, public_search_output]);
    let second = run(
        request(&second_input, &json!([namespace.clone()])),
        Arc::clone(&context),
    )
    .await;
    let OutputItem::FunctionCall(call) = &second.output[0] else {
        panic!("loaded namespace member remains a client function call")
    };
    assert_eq!(call.namespace.as_deref(), Some(namespace_name));
    assert_eq!(call.name, member_name);
    assert_ne!(call.name, flat_name, "private flat name must not leak publicly");

    let mut third_items = second_input.as_array().expect("history").clone();
    third_items.push(serde_json::to_value(&second.output[0]).expect("namespace call serializes"));
    third_items.push(json!({
        "type": "function_call_output", "call_id": "call_forecast", "output": "sunny"
    }));
    let third = run(request(&Value::Array(third_items), &json!([namespace])), context).await;
    assert!(matches!(&third.output[0], OutputItem::Message(_)));

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[1]["tools"].as_array().map(Vec::len), Some(2));
    assert!(
        requests[1]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == flat_name)
    );
    assert_eq!(requests[2]["input"][3]["name"], flat_name);
    assert!(requests[2]["input"][3].get("namespace").is_none());
}

#[tokio::test]
async fn withheld_function_calls_fail_before_gateway_side_effects() {
    let namespace = weather_namespace();
    let flat_name = model_visible_namespace_member_name(
        namespace["name"].as_str().unwrap(),
        namespace["tools"][1]["name"].as_str().unwrap(),
    );
    for (name, tool) in [("get_weather".to_owned(), deferred_weather()), (flat_name, namespace)] {
        let (llm_url, requests, _server) = spawn_sequenced_llm(vec![json!({
            "id": "upstream_withheld", "object": "response", "status": "completed", "model": "test",
            "created_at": 0, "output": [
                {
                    "type": "function_call", "id": "fc_web", "call_id": "call_web", "name": "web_search",
                    "arguments": "{\"query\":\"weather\"}", "status": "completed"
                },
                {
                    "type": "function_call", "id": "fc_withheld", "call_id": "call_withheld", "name": name,
                    "namespace": null, "arguments": "{}", "status": "completed"
                }
            ]
        })])
        .await;
        let calls = Arc::new(AtomicUsize::new(0));
        let context = Arc::new(
            ExecutionContext::new(
                ConversationHandler::new(ConversationStore::disabled()),
                ResponseHandler::new(ResponseStore::disabled()),
                Arc::new(reqwest::Client::new()),
                llm_url,
            )
            .with_gateway_executor(Arc::new(CountingWebSearch {
                calls: Arc::clone(&calls),
            })),
        );
        let payload = request(
            &json!("find weather"),
            &json!([search_declaration(), tool, {"type": "web_search_preview"}]),
        );

        let Err(error) = Box::pin(ExecuteRequest::new(payload, context).run()).await else {
            panic!("known withheld function call must reject the whole response")
        };
        assert_eq!(error.http_status(), http::StatusCode::BAD_GATEWAY);
        assert_eq!(calls.load(Ordering::SeqCst), 0, "gateway call must not execute");
        assert_eq!(requests.lock().await.len(), 1, "only inference may run");
    }
}

#[tokio::test]
async fn withheld_namespace_history_calls_fail_before_inference() {
    let namespace = weather_namespace();
    let namespace_name = namespace["name"].as_str().expect("namespace name");
    let member_name = namespace["tools"][1]["name"].as_str().expect("member name");
    let flat_name = model_visible_namespace_member_name(namespace_name, member_name);
    let (llm_url, requests, _server) = spawn_sequenced_llm(Vec::new()).await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));

    for call in [
        json!({
            "type": "function_call", "id": "fc_public", "call_id": "call_public",
            "namespace": namespace_name, "name": member_name, "arguments": "{}", "status": "completed"
        }),
        json!({
            "type": "function_call", "id": "fc_flat", "call_id": "call_flat",
            "name": flat_name, "arguments": "{}", "status": "completed"
        }),
    ] {
        let payload = request(&json!([call]), &json!([search_declaration(), namespace.clone()]));
        let Err(error) = Box::pin(ExecuteRequest::new(payload, Arc::clone(&context)).run()).await else {
            panic!("withheld history call must fail before inference")
        };
        assert_eq!(error.http_status(), http::StatusCode::BAD_REQUEST);
    }
    assert!(requests.lock().await.is_empty(), "inference must not run");
}

#[tokio::test]
async fn dynamic_namespace_forward_references_fail_before_inference() {
    let namespace = weather_namespace();
    let namespace_name = namespace["name"].as_str().expect("namespace name");
    let member_name = namespace["tools"][1]["name"].as_str().expect("member name");
    let flat_name = model_visible_namespace_member_name(namespace_name, member_name);
    let loaded_subset = loaded_weather_namespace_subset();
    let (llm_url, requests, _server) = spawn_sequenced_llm(Vec::new()).await;
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));

    for call in [
        json!({
            "type": "function_call", "id": "fc_public", "call_id": "call_public",
            "namespace": namespace_name, "name": member_name, "arguments": "{}", "status": "completed"
        }),
        json!({
            "type": "function_call", "id": "fc_flat", "call_id": "call_flat",
            "name": flat_name, "arguments": "{}", "status": "completed"
        }),
    ] {
        let payload = request(
            &json!([
                call,
                {
                    "type": "tool_search_call", "id": "tsc_dynamic", "call_id": "call_dynamic",
                    "arguments": {"query": "forecast"}
                },
                {
                    "type": "tool_search_output", "call_id": "call_dynamic",
                    "tools": [loaded_subset.clone()]
                }
            ]),
            &json!([search_declaration()]),
        );
        let Err(error) = Box::pin(ExecuteRequest::new(payload, Arc::clone(&context)).run()).await else {
            panic!("namespace forward reference must fail before inference")
        };
        assert_eq!(error.http_status(), http::StatusCode::BAD_REQUEST);
    }
    assert!(requests.lock().await.is_empty(), "inference must not run");
}

fn visit_recorded_values(value: &Value, visitor: &mut impl FnMut(&serde_json::Map<String, Value>)) {
    match value {
        Value::Object(object) => {
            visitor(object);
            for child in object.values() {
                visit_recorded_values(child, visitor);
            }
        }
        Value::Array(array) => {
            for child in array {
                visit_recorded_values(child, visitor);
            }
        }
        Value::String(text) => {
            if matches!(text.trim().as_bytes().first(), Some(b'{' | b'['))
                && let Ok(decoded) = serde_json::from_str::<Value>(text)
            {
                visit_recorded_values(&decoded, visitor);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn validate_gateway_cassette_has_no_private_search_projection(raw: &Value) -> Result<(), String> {
    let mut public_search_call_ids = HashSet::new();
    let mut search_item_ids = HashSet::new();
    visit_recorded_values(raw, &mut |object| {
        let item_type = object.get("type").and_then(Value::as_str);
        if item_type == Some("tool_search_call") {
            if let Some(call_id) = object.get("call_id").and_then(Value::as_str) {
                public_search_call_ids.insert(call_id.to_owned());
            }
            if let Some(item_id) = object.get("id").and_then(Value::as_str) {
                search_item_ids.insert(item_id.to_owned());
            }
        } else if item_type == Some("function_call")
            && object.get("name").and_then(Value::as_str) == Some("tool_search")
            && let Some(item_id) = object.get("id").and_then(Value::as_str)
        {
            search_item_ids.insert(item_id.to_owned());
        }
    });

    let mut violation = None;
    visit_recorded_values(raw, &mut |object| {
        let item_type = object.get("type").and_then(Value::as_str);
        let name = object.get("name").and_then(Value::as_str);
        if matches!(item_type, Some("function" | "function_call")) && name == Some("tool_search") {
            violation.get_or_insert_with(|| {
                format!("public gateway cassette leaked the private synthetic search projection: {object:?}")
            });
        } else if item_type == Some("function_call_output") {
            let call_id = object.get("call_id").and_then(Value::as_str);
            if call_id.is_some_and(|call_id| public_search_call_ids.contains(call_id)) {
                violation.get_or_insert_with(|| {
                    format!("public gateway cassette leaked a normalized search function output: {object:?}")
                });
            }
        } else if matches!(
            item_type,
            Some("response.function_call_arguments.delta" | "response.function_call_arguments.done")
        ) {
            let item_id = object.get("item_id").and_then(Value::as_str);
            if item_id.is_some_and(|item_id| search_item_ids.contains(item_id)) {
                violation.get_or_insert_with(|| {
                    format!("public gateway cassette leaked normalized search argument events: {object:?}")
                });
            }
        }
    });
    violation.map_or(Ok(()), Err)
}

#[test]
fn provider_parity_matrix_is_exact_and_gateway_has_no_private_search_leaks() {
    const FLOW_CASSETTES: [(&str, bool); 7] = [
        ("tool-search-openai-reference-gpt-5.6-nonstreaming.yaml", false),
        ("tool-search-openai-reference-gpt-5.6-streaming.yaml", false),
        (
            "tool-search-direct-vllm-Qwen-Qwen3.6-35B-A3B-FP8-nonstreaming.yaml",
            false,
        ),
        ("tool-search-direct-vllm-Qwen-Qwen3.6-35B-A3B-FP8-streaming.yaml", false),
        ("tool-search-gateway-Qwen-Qwen3.6-35B-A3B-FP8-nonstreaming.yaml", true),
        ("tool-search-gateway-Qwen-Qwen3.6-35B-A3B-FP8-streaming.yaml", true),
        ("tool-search-gateway-Qwen-Qwen3.6-35B-A3B-FP8-websocket.yaml", true),
    ];

    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cassettes/tool_search");
    let expected_names = FLOW_CASSETTES
        .iter()
        .map(|(filename, _)| (*filename).to_owned())
        .collect::<HashSet<_>>();
    let actual_names = fs::read_dir(&directory)
        .expect("tool-search cassette directory")
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                return None;
            }
            let cassette = support::load_cassette(path.to_str().expect("cassette path"));
            (cassette.turns.len() == 3).then(|| {
                path.file_name()
                    .and_then(|filename| filename.to_str())
                    .expect("cassette filename")
                    .to_owned()
            })
        })
        .collect::<HashSet<_>>();
    assert_eq!(
        actual_names, expected_names,
        "the final seven-cassette flow matrix must be exact"
    );

    for (filename, gateway) in FLOW_CASSETTES {
        let path = directory.join(filename);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                fs::metadata(&path).expect("cassette metadata").permissions().mode() & 0o111,
                0,
                "checked-in cassette must not be executable: {filename}"
            );
        }
        if gateway {
            let raw =
                serde_yaml::from_str::<Value>(&fs::read_to_string(&path).expect("gateway cassette should be readable"))
                    .expect("gateway cassette YAML");
            let raw_turns = raw["turns"].as_array().expect("raw gateway cassette turns");
            let cassette = support::load_cassette(path.to_str().expect("gateway cassette path"));
            let mut decoded_surfaces = Vec::new();
            for (raw_turn, turn) in raw_turns.iter().zip(&cassette.turns) {
                decoded_surfaces.push(raw_turn["request"]["body"].clone());
                if let Some(body) = &turn.response.body {
                    decoded_surfaces.push(body.clone());
                }
                if turn.response.sse.is_some() {
                    decoded_surfaces.extend(support::recorded_named_sse_events(turn));
                }
                if let Some(websocket) = raw_turn["response"]["websocket"].as_array() {
                    decoded_surfaces.extend(websocket.iter().cloned());
                }
            }
            validate_gateway_cassette_has_no_private_search_projection(&Value::Array(decoded_surfaces))
                .unwrap_or_else(|error| panic!("{error}"));
        }
    }
}

#[test]
fn provider_parity_non_leak_detector_rejects_nested_private_shapes() {
    let cases = [
        json!({"response": {"tools": [{"type": "function", "name": "tool_search"}]}}),
        json!({"response": {"output": [{
            "type": "function_call", "id": "fc_private", "call_id": "call_search", "name": "tool_search"
        }]}}),
        json!({
            "request": {"input": [{
                "type": "tool_search_call", "id": "tsc_public", "call_id": "call_search"
            }, {
                "type": "function_call_output", "call_id": "call_search", "output": "{}"
            }]}
        }),
        json!([{
            "type": "response.output_item.added",
            "item": {"type": "tool_search_call", "id": "tsc_public", "call_id": "call_search"}
        }, {
            "type": "response.function_call_arguments.delta", "item_id": "tsc_public", "delta": "{}"
        }]),
        json!([r#"{"type":"function_call","id":"fc_ws","call_id":"call_ws","name":"tool_search"}"#]),
    ];

    for case in cases {
        assert!(
            validate_gateway_cassette_has_no_private_search_projection(&case).is_err(),
            "private nested shape should be rejected: {case}"
        );
    }
}
