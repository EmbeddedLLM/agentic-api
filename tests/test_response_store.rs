mod common;
#[path = "common/store.rs"]
mod store;

use store::{make_request, make_response};

use agentic_api::store::response::ResponseStore;
use agentic_api::store::translator::InOutItem;

#[tokio::test]
async fn test_get_returns_none_for_missing() {
    let pool = common::create_test_pool().await;
    let store = ResponseStore::new(pool);
    assert!(store.get("nonexistent_id").await.unwrap().is_none());
}

#[tokio::test]
async fn test_get_or_raise_errors_for_missing() {
    let pool = common::create_test_pool().await;
    let store = ResponseStore::new(pool);
    assert!(store.get_or_raise("nonexistent_id").await.is_err());
}

#[tokio::test]
async fn test_put_and_get_round_trip() {
    let pool = common::create_test_pool().await;
    let store = ResponseStore::new(pool);
    let request = make_request("gpt-4o");
    let response = make_response("resp_abc", "gpt-4o", "completed");

    store.put_completed(&request, &request, &response).await.unwrap();

    let stored = store.get("resp_abc").await.unwrap().unwrap();
    assert_eq!(stored.response_id, "resp_abc");
    assert_eq!(stored.metadata.model, "gpt-4o");
    assert!(!stored.history_item_ids.is_empty());
}

#[tokio::test]
async fn test_put_skipped_when_store_disabled() {
    let pool = common::create_test_pool().await;
    let store = ResponseStore::new(pool);
    let mut request = make_request("gpt-4o");
    request.store = false;
    let response = make_response("resp_skip", "gpt-4o", "completed");

    store.put_completed(&request, &request, &response).await.unwrap();
    assert!(store.get("resp_skip").await.unwrap().is_none());
}

#[tokio::test]
async fn test_put_skipped_when_status_not_persistable() {
    let pool = common::create_test_pool().await;
    let store = ResponseStore::new(pool);
    let request = make_request("gpt-4o");
    let response = make_response("resp_failed", "gpt-4o", "failed");

    store.put_completed(&request, &request, &response).await.unwrap();
    assert!(store.get("resp_failed").await.unwrap().is_none());
}

#[tokio::test]
async fn test_rehydrate_restores_items_in_order() {
    let pool = common::create_test_pool().await;
    let store = ResponseStore::new(pool);
    let request = make_request("gpt-4o");
    let response = make_response("resp_rehydrate", "gpt-4o", "completed");

    store.put_completed(&request, &request, &response).await.unwrap();

    let stored = store.get("resp_rehydrate").await.unwrap().unwrap();
    let items = store.rehydrate(&stored).await.unwrap();

    assert!(!items.is_empty());
    assert!(items.iter().any(|i| matches!(i, InOutItem::Output(_))));
}

#[tokio::test]
async fn test_previous_response_id_stored() {
    let pool = common::create_test_pool().await;
    let store = ResponseStore::new(pool);

    let request1 = make_request("gpt-4o");
    let response1 = make_response("resp_turn1", "gpt-4o", "completed");
    store.put_completed(&request1, &request1, &response1).await.unwrap();

    let mut request2 = make_request("gpt-4o");
    request2.previous_response_id = Some("resp_turn1".to_string());
    let mut response2 = make_response("resp_turn2", "gpt-4o", "completed");
    response2.previous_response_id = Some("resp_turn1".to_string());
    store.put_completed(&request2, &request2, &response2).await.unwrap();

    let stored = store.get("resp_turn2").await.unwrap().unwrap();
    assert_eq!(stored.metadata.previous_response_id, Some("resp_turn1".to_string()));
}
