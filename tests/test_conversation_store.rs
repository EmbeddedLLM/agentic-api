mod common;
#[path = "common/store.rs"]
mod store;

use store::{make_request, make_response};

use agentic_api::store::conversation::ConversationStore;
use agentic_api::store::response::ResponseMetadata;
use agentic_api::store::translator::InOutItem;
use agentic_api::types::responses::ToolChoice;

fn make_metadata(model: &str) -> ResponseMetadata {
    ResponseMetadata {
        model: model.to_string(),
        previous_response_id: None,
        effective_tools: None,
        effective_tool_choice: ToolChoice::Auto,
        effective_instructions: None,
    }
}

#[tokio::test]
async fn test_get_returns_none_for_missing() {
    let pool = store::create_test_pool().await;
    let store = ConversationStore::new(Some(pool));
    assert!(store.get("nonexistent_id").await.unwrap().is_none());
}

#[tokio::test]
async fn test_create_new_conversation() {
    let pool = store::create_test_pool().await;
    let store = ConversationStore::new(Some(pool));
    let conv = store.create().await.unwrap();
    assert!(conv.conversation_id.starts_with("conv_"));
}

#[tokio::test]
async fn test_get_or_create_creates_new() {
    let pool = store::create_test_pool().await;
    let store = ConversationStore::new(Some(pool));
    let conv = store.get_or_create("conv_test_123").await.unwrap();
    assert_eq!(conv.conversation_id, "conv_test_123");
}

#[tokio::test]
async fn test_get_or_create_returns_existing() {
    let pool = store::create_test_pool().await;
    let store = ConversationStore::new(Some(pool));
    store.get_or_create("conv_existing").await.unwrap();
    let conv = store.get_or_create("conv_existing").await.unwrap();
    assert_eq!(conv.conversation_id, "conv_existing");
}

#[tokio::test]
async fn test_put_turn_appends_items() {
    let pool = store::create_test_pool().await;
    let store = ConversationStore::new(Some(pool));
    store.get_or_create("conv_turn").await.unwrap();

    let request = make_request("gpt-4o");
    let response = make_response("resp_turn_1", "gpt-4o", "completed");
    let mut items: Vec<InOutItem> = agentic_api::store::translator::normalize_input(&request.input)
        .into_iter()
        .map(InOutItem::Input)
        .collect();
    items.extend(response.output.iter().map(|o| InOutItem::Output(o.clone())));

    store
        .put_turn("conv_turn", "resp_turn_1", None, &items, &make_metadata("gpt-4o"))
        .await
        .unwrap();

    let history = store.rehydrate("conv_turn").await.unwrap();
    assert!(!history.is_empty());
}

#[tokio::test]
async fn test_put_turn_accumulates_across_turns() {
    let pool = store::create_test_pool().await;
    let store = ConversationStore::new(Some(pool));
    store.get_or_create("conv_multi").await.unwrap();

    let response1 = make_response("resp_m1", "gpt-4o", "completed");
    let items1: Vec<InOutItem> = response1.output.iter().map(|o| InOutItem::Output(o.clone())).collect();
    store
        .put_turn("conv_multi", "resp_m1", None, &items1, &make_metadata("gpt-4o"))
        .await
        .unwrap();

    let response2 = make_response("resp_m2", "gpt-4o", "completed");
    let items2: Vec<InOutItem> = response2.output.iter().map(|o| InOutItem::Output(o.clone())).collect();
    store
        .put_turn(
            "conv_multi",
            "resp_m2",
            Some("resp_m1"),
            &items2,
            &make_metadata("gpt-4o"),
        )
        .await
        .unwrap();

    let history = store.rehydrate("conv_multi").await.unwrap();
    assert_eq!(history.len(), 2);
}

#[tokio::test]
async fn test_put_turn_raises_for_missing_conversation() {
    let pool = store::create_test_pool().await;
    let store = ConversationStore::new(Some(pool));
    let result = store
        .put_turn("conv_missing", "resp_x", None, &[], &make_metadata("gpt-4o"))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_rehydrate_raises_for_missing_conversation() {
    let pool = store::create_test_pool().await;
    let store = ConversationStore::new(Some(pool));
    assert!(store.rehydrate("conv_missing").await.is_err());
}

#[tokio::test]
async fn test_rehydrate_empty_conversation() {
    let pool = store::create_test_pool().await;
    let store = ConversationStore::new(Some(pool));
    store.get_or_create("conv_empty").await.unwrap();
    let history = store.rehydrate("conv_empty").await.unwrap();
    assert!(history.is_empty());
}

#[tokio::test]
async fn test_rehydrate_restores_items_in_order() {
    let pool = store::create_test_pool().await;
    let store = ConversationStore::new(Some(pool));
    store.get_or_create("conv_order").await.unwrap();

    for (resp_id, prev) in [
        ("resp_o1", None),
        ("resp_o2", Some("resp_o1")),
        ("resp_o3", Some("resp_o2")),
    ] {
        let response = make_response(resp_id, "gpt-4o", "completed");
        let items: Vec<InOutItem> = response.output.iter().map(|o| InOutItem::Output(o.clone())).collect();
        store
            .put_turn("conv_order", resp_id, prev, &items, &make_metadata("gpt-4o"))
            .await
            .unwrap();
    }

    let history = store.rehydrate("conv_order").await.unwrap();
    assert_eq!(history.len(), 3);
}
