//! What the model sees of a hosted tool search, and what is stored for it.
//!
//! A live round, a `previous_response_id` continuation and a `store: false` replay must show the
//! model the same call and output, so later prompts share the live round's prefix. Multi-agent
//! histories stored before the public pair became the canonical record hold the model's private
//! pair; `tests/fixtures/multi_agent_private_search_history.json` is one, captured from this code.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agentic_core::executor::{ConversationHandler, ExecuteRequest, ExecutionContext, ResponseHandler};
use agentic_core::storage::{ConversationStore, ResponseStore, create_pool_with_schema};
use agentic_core::types::request_response::{RequestPayload, ResponsePayload};
use axum::Router;
use axum::body::Bytes;
use axum::response::IntoResponse;
use axum::routing::post;
use either::Either;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

type Requests = Arc<Mutex<Vec<Value>>>;

const PRIVATE_HISTORY_FIXTURE: &str = "tests/fixtures/multi_agent_private_search_history.json";

/// A mock upstream that answers each request with the next prepared JSON body.
async fn spawn_llm(responses: Vec<Value>) -> (String, Requests) {
    let responses = Arc::new(Mutex::new(VecDeque::from(responses)));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let (route_responses, route_requests) = (Arc::clone(&responses), Arc::clone(&requests));
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let (responses, requests) = (Arc::clone(&route_responses), Arc::clone(&route_requests));
            async move {
                requests
                    .lock()
                    .await
                    .push(serde_json::from_slice(&body).expect("request JSON"));
                let response = responses.lock().await.pop_front().expect("one response per request");
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
    tokio::spawn(async move { axum::serve(listener, app).await.expect("mock server") });
    (format!("http://{address}"), requests)
}

async fn durable_context(llm_url: String) -> (Arc<ExecutionContext>, ResponseStore) {
    static DATABASE: AtomicU64 = AtomicU64::new(0);
    let url = format!(
        "sqlite:file:hosted_tool_search_projection_{}?mode=memory&cache=shared",
        DATABASE.fetch_add(1, Ordering::Relaxed)
    );
    let pool = create_pool_with_schema(Some(&url)).await.expect("storage pool");
    let responses = ResponseStore::new(Arc::clone(&pool));
    let context = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::new(pool)),
        ResponseHandler::new(responses.clone()),
        Arc::new(reqwest::Client::new()),
        llm_url,
    ));
    (context, responses)
}

fn upstream(id: &str, output: Value) -> Value {
    let mut response = json!({"id": id, "object": "response", "status": "completed", "model": "test", "created_at": 0});
    response["output"] = output;
    response
}

fn search_call(suffix: &str) -> Value {
    json!({"type": "function_call", "id": format!("fc_{suffix}"), "call_id": format!("call_{suffix}"),
        "name": "tool_search", "arguments": json!({"paths": ["get_weather"]}).to_string(), "status": "completed"})
}

fn message(text: &str) -> Value {
    json!({"type": "message", "id": format!("msg_{text}"), "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}]})
}

fn user(text: &str) -> Value {
    json!({"type": "message", "role": "user", "content": text})
}

fn tools() -> Value {
    json!([{"type": "tool_search"}, {"type": "function", "name": "get_weather", "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
            "additionalProperties": false},
        "strict": true, "defer_loading": true}])
}

async fn run(body: Value, context: &Arc<ExecutionContext>) -> ResponsePayload {
    let request: RequestPayload = serde_json::from_value(body).expect("public request");
    match Box::pin(ExecuteRequest::new(request, Arc::clone(context)).run())
        .await
        .expect("blocking execution")
    {
        Either::Left(response) => response,
        Either::Right(_) => panic!("blocking request returned a stream"),
    }
}

fn model_input(request: &Value) -> Vec<Value> {
    request["input"].as_array().expect("item input").clone()
}

#[tokio::test]
async fn live_continuation_and_replay_show_the_model_the_same_search() {
    let (llm_url, requests) = spawn_llm(vec![
        upstream("up_1", json!([search_call("s1")])),
        upstream("up_2", json!([message("done")])),
        upstream("up_3", json!([message("continued")])),
        upstream("up_4", json!([message("replayed")])),
    ])
    .await;
    let (context, _) = durable_context(llm_url).await;
    let first = run(
        json!({"model": "test", "input": [user("weather?")], "tools": tools(), "store": true}),
        &context,
    )
    .await;
    run(
        json!({"model": "test", "input": [user("more")], "tools": tools(), "store": true,
            "previous_response_id": first.id}),
        &context,
    )
    .await;
    let mut replay = vec![user("weather?")];
    replay.extend(
        first
            .output
            .iter()
            .map(|item| serde_json::to_value(item).expect("output item")),
    );
    replay.push(user("more"));
    run(
        json!({"model": "test", "input": replay, "tools": tools(), "store": false}),
        &context,
    )
    .await;

    let requests = requests.lock().await;
    let live = model_input(&requests[1]);
    assert_eq!(live.len(), 3, "the question, the search call and its output: {live:?}");
    assert_eq!(live[1]["name"], "tool_search");
    assert_eq!(live[2]["type"], "function_call_output");
    assert_eq!(live[1]["call_id"], live[2]["call_id"]);
    for (path, request) in [("continuation", &requests[2]), ("store: false replay", &requests[3])] {
        let input = model_input(request);
        assert_eq!(input[..live.len()], live[..], "{path} shares the live round's prefix");
    }
}

#[tokio::test]
async fn multi_agent_history_holds_the_models_private_search_pair() {
    let (llm_url, _) = spawn_llm(vec![
        upstream("up_1", json!([search_call("s1")])),
        upstream("up_2", json!([message("done")])),
    ])
    .await;
    let (context, responses) = durable_context(llm_url).await;
    let response = run(
        json!({"model": "test", "input": "weather?", "store": true, "multi_agent": {"enabled": true},
            "tools": tools()}),
        &context,
    )
    .await;
    let stored = responses.get(&response.id).await.expect("stored response");
    let tree = stored.metadata.multi_agent_tree.expect("multi-agent tree");
    let history = serde_json::to_value(&tree.agents[0].history).expect("history JSON");
    let items = history.as_array().expect("history items");
    let call = items
        .iter()
        .find(|item| item["type"] == "function_call" && item["name"] == "tool_search")
        .expect("the model's search call");
    assert!(
        items
            .iter()
            .any(|item| item["type"] == "function_call_output" && item["call_id"] == call["call_id"]),
        "{history:#}"
    );
    assert!(
        !items
            .iter()
            .any(|item| item["type"] == "tool_search_call" || item["type"] == "tool_search_output"),
        "{history:#}"
    );
    if std::env::var_os("AGENTIC_WRITE_FIXTURES").is_some() {
        let text = serde_json::to_string_pretty(&history).expect("fixture JSON");
        std::fs::write(PRIVATE_HISTORY_FIXTURE, text + "\n").expect("write fixture");
    }
}

/// The summary request carries no tools, yet the model still reads the search as its private
/// pair and the searched function's call without its own name as namespace (tjtanaa's review).
#[tokio::test]
async fn compaction_summary_shows_the_search_and_its_call_as_the_model_saw_them() {
    let (llm_url, requests) = spawn_llm(vec![
        upstream("up_summary", json!([message("summary")])),
        upstream("up_1", json!([message("done")])),
    ])
    .await;
    let (context, _) = durable_context(llm_url).await;
    let input = json!([
        user("weather?"),
        {"type": "tool_search_call", "id": "tsc_s1", "call_id": null, "execution": "server",
            "arguments": {"paths": ["get_weather"]}, "status": "completed"},
        {"type": "tool_search_output", "id": "tso_s1", "call_id": null, "execution": "server", "status": "completed",
            "tools": [tools()[1].clone()]},
        {"type": "function_call", "id": "fc_w1", "call_id": "call_w1", "name": "get_weather",
            "namespace": "get_weather", "arguments": "{\"city\":\"Paris\"}", "status": "completed"},
        {"type": "function_call_output", "call_id": "call_w1", "output": "18C"},
        user("again")
    ]);
    let response = run(
        json!({"model": "test", "input": input, "tools": tools(), "store": false,
            "context_management": [{"type": "compaction", "compact_threshold": 1}]}),
        &context,
    )
    .await;
    assert_eq!(response.status, "completed");

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 2, "a summary request, then the answer");
    assert!(requests[0]["tools"].is_null(), "{}", requests[0]["tools"]);
    let summary = model_input(&requests[0]);
    assert_eq!(summary[1]["type"], "function_call");
    assert_eq!(summary[1]["name"], "tool_search");
    assert_eq!(summary[1]["call_id"], "tsc_s1");
    assert_eq!(summary[2]["type"], "function_call_output");
    assert_eq!(summary[2]["call_id"], "tsc_s1");
    assert_eq!(summary[3]["name"], "get_weather");
    assert!(summary[3].get("namespace").is_none(), "{}", summary[3]);
}
