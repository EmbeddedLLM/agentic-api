//! HTTP transport for conversation item CRUD.

use axum::extract::{Path, Query, Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use agentic_core::storage::{InOutItem, Item};
#[cfg(feature = "openapi")]
use agentic_core::types::ConversationResponse;
use agentic_core::types::conversations::ItemOrder;
use agentic_core::types::{ConversationItem, CreateItemRequest, ItemResponse, ListItemsResponse};

use super::super::common::{error_response, extract_json, read_bytes};
use super::conversations::{conversation_response, extract_tenant_id, storage_error};
use crate::app::AppState;

/// Query parameters for listing conversation items.
#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::IntoParams))]
pub struct ListItemsQuery {
    #[serde(default = "default_limit")]
    pub limit: u8,
    pub after: Option<String>,
    #[serde(default)]
    pub order: ItemOrder,
}

fn default_limit() -> u8 {
    20
}

#[cfg_attr(feature = "openapi", utoipa::path(
    post,
    path = "/v1/conversations/{conversation_id}/items",
    params(
        ("conversation_id" = String, Path, description = "Conversation ID")
    ),
    request_body = CreateItemRequest,
    responses(
        (status = 200, description = "Items created", body = ListItemsResponse),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Conversation not found"),
    ),
    security(("bearer_auth" = [])),
    tag = "conversations",
))]
pub async fn create_item(State(state): State<AppState>, Path(id): Path<String>, req: Request) -> Response {
    let tenant = match extract_tenant_id(&req) {
        Ok(id) => id,
        Err(error) => return error,
    };
    let bytes = match read_bytes(req.into_body(), state.max_request_body_size).await {
        Ok(bytes) => bytes,
        Err(error) => return error,
    };
    let request: CreateItemRequest = match extract_json(&bytes) {
        Ok(request) => request,
        Err(error) => return error,
    };
    if request.items.is_empty() || request.items.len() > 20 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "items must contain between 1 and 20 items",
        );
    }
    let items = request.items.into_iter().map(into_stored_item).collect();
    match state
        .exec_ctx
        .conv_handler
        .store()
        .create_items(&tenant, &id, items)
        .await
    {
        Ok(items) => list_response(items, false),
        Err(error) => storage_error(error),
    }
}

#[cfg_attr(feature = "openapi", utoipa::path(
    get,
    path = "/v1/conversations/{conversation_id}/items",
    params(
        ("conversation_id" = String, Path, description = "Conversation ID"),
        ListItemsQuery
    ),
    responses(
        (status = 200, description = "Items retrieved", body = ListItemsResponse),
        (status = 404, description = "Conversation not found"),
    ),
    security(("bearer_auth" = [])),
    tag = "conversations",
))]
pub async fn list_items(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<ListItemsQuery>,
    req: Request,
) -> Response {
    let tenant = match extract_tenant_id(&req) {
        Ok(id) => id,
        Err(error) => return error,
    };
    if !(1..=100).contains(&query.limit) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "limit must be between 1 and 100",
        );
    }
    let limit = i64::from(query.limit);
    match state
        .exec_ctx
        .conv_handler
        .store()
        .list_items(&tenant, &id, limit + 1, query.after.as_deref(), query.order)
        .await
    {
        Ok(mut items) => {
            let has_more = items.len() > usize::from(query.limit);
            items.truncate(usize::from(query.limit));
            list_response(items, has_more)
        }
        Err(error) => storage_error(error),
    }
}

#[cfg_attr(feature = "openapi", utoipa::path(
    get,
    path = "/v1/conversations/{conversation_id}/items/{item_id}",
    params(
        ("conversation_id" = String, Path, description = "Conversation ID"),
        ("item_id" = String, Path, description = "Item ID")
    ),
    responses(
        (status = 200, description = "Item retrieved", body = ItemResponse),
        (status = 404, description = "Item or conversation not found"),
    ),
    security(("bearer_auth" = [])),
    tag = "conversations",
))]
pub async fn retrieve_item(
    State(state): State<AppState>,
    Path((id, item_id)): Path<(String, String)>,
    req: Request,
) -> Response {
    let tenant = match extract_tenant_id(&req) {
        Ok(id) => id,
        Err(error) => return error,
    };
    match state
        .exec_ctx
        .conv_handler
        .store()
        .retrieve_item(&tenant, &id, &item_id)
        .await
    {
        Ok(item) => match item_response(item) {
            Ok(item) => axum::Json(item).into_response(),
            Err(error) => error,
        },
        Err(error) => storage_error(error),
    }
}

#[cfg_attr(feature = "openapi", utoipa::path(
    delete,
    path = "/v1/conversations/{conversation_id}/items/{item_id}",
    params(
        ("conversation_id" = String, Path, description = "Conversation ID"),
        ("item_id" = String, Path, description = "Item ID")
    ),
    responses(
        (status = 200, description = "Item deleted", body = ConversationResponse),
        (status = 404, description = "Item or conversation not found"),
    ),
    security(("bearer_auth" = [])),
    tag = "conversations",
))]
pub async fn delete_item(
    State(state): State<AppState>,
    Path((id, item_id)): Path<(String, String)>,
    req: Request,
) -> Response {
    let tenant = match extract_tenant_id(&req) {
        Ok(id) => id,
        Err(error) => return error,
    };
    match state
        .exec_ctx
        .conv_handler
        .store()
        .delete_item(&tenant, &id, &item_id)
        .await
    {
        Ok(conversation) => conversation_response(conversation),
        Err(error) => storage_error(error),
    }
}

pub(super) fn into_stored_item(item: ConversationItem) -> InOutItem {
    match item {
        ConversationItem::Input(input) => InOutItem::Input(input),
        ConversationItem::Output(output) => InOutItem::Output(output),
    }
}

#[allow(clippy::result_large_err)]
fn item_response(item: Item) -> Result<ItemResponse, Response> {
    let content = match item.as_inout() {
        Some(InOutItem::Input(input)) => ConversationItem::Input(input),
        Some(InOutItem::Output(output)) => ConversationItem::Output(output),
        None => {
            return Err(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                "failed to decode stored item",
            ));
        }
    };
    Ok(ItemResponse::new(item.id, content))
}

fn list_response(items: Vec<Item>, has_more: bool) -> Response {
    match items.into_iter().map(item_response).collect::<Result<Vec<_>, _>>() {
        Ok(items) => axum::Json(ListItemsResponse::new(items, has_more)).into_response(),
        Err(error) => error,
    }
}
