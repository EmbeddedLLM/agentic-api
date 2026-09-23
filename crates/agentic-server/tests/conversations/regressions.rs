use super::*;
use agentic_core::types::conversations::ItemOrder;

use agentic_core::storage::models::item as item_model;
use agentic_core::storage::{InOutItem, ResponseMetadata};
use agentic_core::types::ConversationItem;

fn regression_message(text: &str) -> InOutItem {
    InOutItem::Input(
        serde_json::from_value(json!({
            "type": "message", "role": "user", "content": text
        }))
        .unwrap(),
    )
}

#[tokio::test]
async fn regression_openai_batch_item_request() {
    let state = test_state_with_storage("http://127.0.0.1:1").await;
    let conv = state
        .exec_ctx
        .conv_handler
        .store()
        .create_with_metadata_and_items(Some("default_tenant"), None, vec![])
        .await
        .unwrap();
    let (url, handle) = spawn_gateway(state).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/conversations/{}/items", conv.conversation_id))
        .json(&json!({"items": [{"type":"message", "role":"user", "content":"hello"}]}))
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body = response.text().await.unwrap();
    handle.abort();
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn regression_list_wire_shape() {
    let state = test_state_with_storage("http://127.0.0.1:1").await;
    let conv = state
        .exec_ctx
        .conv_handler
        .store()
        .create_with_metadata_and_items(Some("default_tenant"), None, vec![regression_message("hello")])
        .await
        .unwrap();
    let (url, handle) = spawn_gateway(state).await;
    let body: serde_json::Value = reqwest::Client::new()
        .get(format!("{url}/v1/conversations/{}/items", conv.conversation_id))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    handle.abort();
    assert_eq!(body["data"][0]["type"], "message", "{body}");
}

#[tokio::test]
async fn regression_list_descending_order() {
    let state = test_state_with_storage("http://127.0.0.1:1").await;
    let conv = state
        .exec_ctx
        .conv_handler
        .store()
        .create_with_metadata_and_items(
            Some("default_tenant"),
            None,
            vec![regression_message("first"), regression_message("last")],
        )
        .await
        .unwrap();
    let (url, handle) = spawn_gateway(state).await;
    let body: serde_json::Value = reqwest::Client::new()
        .get(format!(
            "{url}/v1/conversations/{}/items?order=desc&limit=1",
            conv.conversation_id
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    handle.abort();
    assert_eq!(body["data"][0]["content"][0]["text"], "last", "{body}");
}

#[test]
fn regression_output_item_roundtrip() {
    let wire = json!({"type":"web_search_call", "id":"ws_1", "status":"completed",
        "action":{"type":"search", "query":"weather", "queries":["weather"], "sources":[{"url":"https://example.com"}]}});
    let parsed: ConversationItem = serde_json::from_value(wire.clone()).unwrap();
    let actual = serde_json::to_value(parsed).unwrap();
    assert_eq!(actual, wire);
}

#[tokio::test]
async fn regression_response_items_are_visible() {
    let state = test_state_with_storage("http://127.0.0.1:1").await;
    let store = state.exec_ctx.conv_handler.store();
    let conv = store
        .create_with_metadata_and_items(Some("default_tenant"), None, vec![])
        .await
        .unwrap();
    store
        .persist(
            &conv.conversation_id,
            "resp_review",
            None,
            vec![regression_message("from response persistence")],
            &ResponseMetadata::default(),
        )
        .await
        .unwrap();
    assert_eq!(store.rehydrate(&conv.conversation_id).await.unwrap().len(), 1);
    let (url, handle) = spawn_gateway(state).await;
    let body: serde_json::Value = reqwest::Client::new()
        .get(format!("{url}/v1/conversations/{}/items", conv.conversation_id))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    handle.abort();
    assert_eq!(body["data"].as_array().unwrap().len(), 1, "{body}");
}

#[tokio::test]
async fn regression_delete_preserves_items() {
    let state = test_state_with_storage("http://127.0.0.1:1").await;
    let store = state.exec_ctx.conv_handler.store();
    let conv = store
        .create_with_metadata_and_items(Some("default_tenant"), None, vec![regression_message("preserve me")])
        .await
        .unwrap();
    let items = store
        .list_items("default_tenant", &conv.conversation_id, 100, None, ItemOrder::Asc)
        .await
        .unwrap();
    store.delete("default_tenant", &conv.conversation_id).await.unwrap();
    let preserved = item_model::get_items(store.pool().unwrap(), &[items[0].id.clone()])
        .await
        .unwrap();
    assert_eq!(preserved.len(), 1);
}

#[tokio::test]
async fn regression_cursor_follows_sequence() {
    let state = test_state_with_storage("http://127.0.0.1:1").await;
    let store = state.exec_ctx.conv_handler.store();
    let conv = store
        .create_with_metadata_and_items(Some("default_tenant"), None, vec![])
        .await
        .unwrap();
    let data = String::try_from(&regression_message("hello")).unwrap();
    let earlier_id = agentic_core::utils::common::uuid7_str("item_");
    let later_id = agentic_core::utils::common::uuid7_str("item_");
    assert!(earlier_id < later_id);
    // IDs are allocated before awaiting storage, so creation order can differ from insertion order.
    insert_with_ids(
        store.pool().unwrap(),
        &conv.conversation_id,
        vec![(later_id.clone(), data.clone())],
    )
    .await
    .unwrap();
    insert_with_ids(
        store.pool().unwrap(),
        &conv.conversation_id,
        vec![(earlier_id.clone(), data)],
    )
    .await
    .unwrap();
    let page = store
        .list_items("default_tenant", &conv.conversation_id, 1, None, ItemOrder::Asc)
        .await
        .unwrap();
    assert_eq!(page[0].id, later_id);
    let next = store
        .list_items(
            "default_tenant",
            &conv.conversation_id,
            1,
            Some(&page[0].id),
            ItemOrder::Asc,
        )
        .await
        .unwrap();
    assert_eq!(next.len(), 1, "second item skipped despite having the next sequence");
}

#[tokio::test]
async fn regression_delete_item_returns_conversation() {
    let state = test_state_with_storage("http://127.0.0.1:1").await;
    let store = state.exec_ctx.conv_handler.store();
    let conv = store
        .create_with_metadata_and_items(Some("default_tenant"), None, vec![regression_message("hello")])
        .await
        .unwrap();
    let items = store
        .list_items("default_tenant", &conv.conversation_id, 100, None, ItemOrder::Asc)
        .await
        .unwrap();
    let (url, handle) = spawn_gateway(state).await;
    let body: serde_json::Value = reqwest::Client::new()
        .delete(format!(
            "{url}/v1/conversations/{}/items/{}",
            conv.conversation_id, items[0].id
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    handle.abort();
    assert_eq!(body["object"], "conversation", "{body}");
    assert_eq!(body["id"], conv.conversation_id);
}

#[tokio::test]
async fn regression_delete_invalidates_version() {
    let state = test_state_with_storage("http://127.0.0.1:1").await;
    let store = state.exec_ctx.conv_handler.store();
    let conv = store
        .create_with_metadata_and_items(
            Some("default_tenant"),
            None,
            vec![regression_message("first"), regression_message("last")],
        )
        .await
        .unwrap();
    let snapshot = store.rehydrate_snapshot(&conv.conversation_id).await.unwrap();
    let items = store
        .list_items("default_tenant", &conv.conversation_id, 100, None, ItemOrder::Asc)
        .await
        .unwrap();
    store
        .delete_item("default_tenant", &conv.conversation_id, &items[0].id)
        .await
        .unwrap();
    let result = store
        .persist_if_version(
            &conv.conversation_id,
            snapshot.version,
            "resp_stale",
            None,
            vec![regression_message("based on deleted history")],
            &ResponseMetadata::default(),
        )
        .await;
    assert!(
        matches!(
            result,
            Err(agentic_core::storage::StorageError::ConversationConflict { .. })
        ),
        "{result:?}"
    );
}

#[tokio::test]
async fn regression_concurrent_item_creation() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}", dir.path().join("review.db").display());
    let pool = create_pool_with_schema(Some(&url)).await.unwrap();
    let store = ConversationStore::new(Arc::clone(&pool));
    let conv = store
        .create_with_metadata_and_items(Some("default_tenant"), None, vec![])
        .await
        .unwrap();
    let barrier = Arc::new(tokio::sync::Barrier::new(10));
    let mut tasks = Vec::new();
    for idx in 0..10 {
        let pool = Arc::clone(&pool);
        let barrier = Arc::clone(&barrier);
        let id = conv.conversation_id.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            ConversationStore::new(pool)
                .create_items("default_tenant", &id, vec![regression_message(&format!("hello {idx}"))])
                .await
        }));
    }
    let results = futures::future::join_all(tasks).await;
    let errors: Vec<_> = results.into_iter().filter_map(|r| r.unwrap().err()).collect();
    assert!(errors.is_empty(), "{errors:?}");
}

async fn insert_with_ids(
    pool: &agentic_core::storage::DbPool,
    id: &str,
    items: Vec<(String, String)>,
) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;
    agentic_core::storage::models::conversation::lock_in_tx(&mut tx, id).await?;
    item_model::create_in_tx(&mut tx, items, Some(id)).await?;
    tx.commit().await
}

#[tokio::test]
async fn deletion_preserves_stored_response_history_and_invalidates_empty_snapshot() {
    let state = test_state_with_storage("http://127.0.0.1:1").await;
    let store = state.exec_ctx.conv_handler.store();
    let conv = store
        .create_with_metadata_and_items(Some("default_tenant"), None, vec![])
        .await
        .unwrap();
    let empty = store.rehydrate_snapshot(&conv.conversation_id).await.unwrap();
    let inputs = vec![regression_message("keep response history")];
    store
        .persist(
            &conv.conversation_id,
            "resp_kept",
            None,
            inputs.clone(),
            &ResponseMetadata::default(),
        )
        .await
        .unwrap();
    let rows = store
        .list_items("default_tenant", &conv.conversation_id, 100, None, ItemOrder::Asc)
        .await
        .unwrap();
    store
        .delete_item("default_tenant", &conv.conversation_id, &rows[0].id)
        .await
        .unwrap();
    let result = store
        .persist_if_version(
            &conv.conversation_id,
            empty.version,
            "resp_stale_empty",
            None,
            vec![],
            &ResponseMetadata::default(),
        )
        .await;
    assert!(matches!(
        result,
        Err(agentic_core::storage::StorageError::ConversationConflict { .. })
    ));
    let response_store = ResponseStore::new(Arc::new(store.pool().unwrap().clone()));
    assert_eq!(response_store.rehydrate("resp_kept").await.unwrap(), inputs);
    store
        .persist(
            &conv.conversation_id,
            "resp_kept_two",
            None,
            inputs.clone(),
            &ResponseMetadata::default(),
        )
        .await
        .unwrap();
    store.delete("default_tenant", &conv.conversation_id).await.unwrap();
    assert_eq!(response_store.rehydrate("resp_kept_two").await.unwrap(), inputs);
}

#[tokio::test]
async fn item_operations_reject_another_tenant_and_another_conversations_cursor() {
    let state = test_state_with_storage("http://127.0.0.1:1").await;
    let store = state.exec_ctx.conv_handler.store();
    let conv = store
        .create_with_metadata_and_items(Some("tenant_a"), None, vec![regression_message("private")])
        .await
        .unwrap();
    let other = store
        .create_with_metadata_and_items(Some("tenant_a"), None, vec![])
        .await
        .unwrap();
    let items = store
        .list_items("tenant_a", &conv.conversation_id, 100, None, ItemOrder::Asc)
        .await
        .unwrap();
    let id = &items[0].id;
    assert!(
        store
            .retrieve_item("tenant_b", &conv.conversation_id, id)
            .await
            .is_err()
    );
    assert!(store.delete_item("tenant_b", &conv.conversation_id, id).await.is_err());
    assert!(
        store
            .create_items("tenant_b", &conv.conversation_id, vec![regression_message("injected")])
            .await
            .is_err()
    );
    assert!(
        store
            .list_items("tenant_a", &other.conversation_id, 100, Some(id), ItemOrder::Asc)
            .await
            .is_err()
    );
    assert_eq!(store.rehydrate(&conv.conversation_id).await.unwrap().len(), 1);
}

#[tokio::test]
async fn rejects_oversized_batches_and_invalid_limits() {
    let state = test_state_with_storage("http://127.0.0.1:1").await;
    let conv = state
        .exec_ctx
        .conv_handler
        .store()
        .create_with_metadata_and_items(Some("default_tenant"), None, vec![])
        .await
        .unwrap();
    let (url, handle) = spawn_gateway(state).await;
    let client = reqwest::Client::new();
    let items = vec![json!({"role":"user", "content":"hello"}); 21];
    for path in [
        "/v1/conversations".to_owned(),
        format!("/v1/conversations/{}/items", conv.conversation_id),
    ] {
        let response = client
            .post(format!("{url}{path}"))
            .json(&json!({"items": items}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
    for query in ["limit=0", "limit=101", "order=invalid"] {
        let response = client
            .get(format!("{url}/v1/conversations/{}/items?{query}", conv.conversation_id))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
    handle.abort();
}
