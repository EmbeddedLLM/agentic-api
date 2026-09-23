use axum::extract::{Path, Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use agentic_core::executor::ExecutorError;
use agentic_core::storage::{ConversationData, InOutItem, StorageError};
use agentic_core::types::{
    ConversationResponse, CreateConversationRequest, DeletedResponse, UpdateConversationRequest,
};

use super::super::common::{error_response, executor_error_response, extract_json, read_bytes};
use crate::app::AppState;

/// Extract tenant ID from authenticated principal in request extensions.
///
/// Returns Result to support future authentication error handling.
#[allow(clippy::unnecessary_wraps, clippy::result_large_err)]
pub(super) fn extract_tenant_id(_req: &Request) -> Result<String, Response> {
    // For now, return a placeholder until we wire up authentication
    // In production, this would extract from the AuthenticatedPrincipal extension
    // and return Err(response) for authentication failures
    Ok("default_tenant".to_string())
}

/// Create a new conversation with optional metadata and initial items.
#[cfg_attr(feature = "openapi", utoipa::path(
    post,
    path = "/v1/conversations",
    request_body(content = Option<CreateConversationRequest>, content_type = "application/json"),
    responses(
        (status = 200, description = "Conversation created", body = ConversationResponse),
        (status = 400, description = "Invalid request"),
    ),
    security(("bearer_auth" = [])),
    tag = "conversations",
))]
pub async fn create_conversation(State(state): State<AppState>, req: Request) -> Response {
    let tenant_id = match extract_tenant_id(&req) {
        Ok(id) => id,
        Err(err) => return err,
    };

    let (_, body) = req.into_parts();
    let bytes = match read_bytes(body, state.max_request_body_size).await {
        Ok(b) => b,
        Err(e) => return e,
    };

    // Empty body is valid - treat as default request with no metadata or items
    let request: CreateConversationRequest = if bytes.is_empty() {
        CreateConversationRequest {
            metadata: None,
            items: None,
        }
    } else {
        match extract_json(&bytes) {
            Ok(r) => r,
            Err(e) => return e,
        }
    };

    if request.items.as_ref().is_some_and(|items| items.len() > 20) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "a conversation accepts at most 20 initial items",
        );
    }

    let initial_items: Vec<InOutItem> = request
        .items
        .unwrap_or_default()
        .into_iter()
        .map(|item| match item {
            agentic_core::types::ConversationItem::Input(input) => InOutItem::Input(input),
            agentic_core::types::ConversationItem::Output(output) => InOutItem::Output(output),
        })
        .collect();

    match state
        .exec_ctx
        .conv_handler
        .store()
        .create_with_metadata_and_items(Some(&tenant_id), request.metadata, initial_items)
        .await
    {
        Ok(data) => conversation_response(data),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("Failed to create conversation: {e}"),
        ),
    }
}

/// Retrieve a conversation by ID.
#[cfg_attr(feature = "openapi", utoipa::path(
    get,
    path = "/v1/conversations/{conversation_id}",
    params(
        ("conversation_id" = String, Path, description = "Conversation ID")
    ),
    responses(
        (status = 200, description = "Conversation retrieved", body = ConversationResponse),
        (status = 404, description = "Conversation not found"),
    ),
    security(("bearer_auth" = [])),
    tag = "conversations",
))]
pub async fn retrieve_conversation(
    State(state): State<AppState>,
    Path(conversation_id): Path<String>,
    req: Request,
) -> Response {
    let tenant_id = match extract_tenant_id(&req) {
        Ok(id) => id,
        Err(err) => return err,
    };

    match state
        .exec_ctx
        .conv_handler
        .store()
        .retrieve(&tenant_id, &conversation_id)
        .await
    {
        Ok(data) => conversation_response(data),
        Err(agentic_core::storage::StorageError::NotFound { .. }) => {
            error_response(StatusCode::NOT_FOUND, "not_found", "Conversation not found")
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("Failed to retrieve conversation: {e}"),
        ),
    }
}

/// Update a conversation's metadata.
#[cfg_attr(feature = "openapi", utoipa::path(
    post,
    path = "/v1/conversations/{conversation_id}",
    params(
        ("conversation_id" = String, Path, description = "Conversation ID")
    ),
    request_body = UpdateConversationRequest,
    responses(
        (status = 200, description = "Conversation updated", body = ConversationResponse),
        (status = 404, description = "Conversation not found"),
    ),
    security(("bearer_auth" = [])),
    tag = "conversations",
))]
pub async fn update_conversation(
    State(state): State<AppState>,
    Path(conversation_id): Path<String>,
    req: Request,
) -> Response {
    let tenant_id = match extract_tenant_id(&req) {
        Ok(id) => id,
        Err(err) => return err,
    };

    let (_, body) = req.into_parts();
    let bytes = match read_bytes(body, state.max_request_body_size).await {
        Ok(b) => b,
        Err(e) => return e,
    };

    let request: UpdateConversationRequest = match extract_json(&bytes) {
        Ok(r) => r,
        Err(e) => return e,
    };

    match state
        .exec_ctx
        .conv_handler
        .store()
        .update_metadata(&tenant_id, &conversation_id, request.metadata)
        .await
    {
        Ok(data) => conversation_response(data),
        Err(agentic_core::storage::StorageError::NotFound { .. }) => {
            error_response(StatusCode::NOT_FOUND, "not_found", "Conversation not found")
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("Failed to update conversation: {e}"),
        ),
    }
}

/// Delete a conversation by ID.
#[cfg_attr(feature = "openapi", utoipa::path(
    delete,
    path = "/v1/conversations/{conversation_id}",
    params(
        ("conversation_id" = String, Path, description = "Conversation ID")
    ),
    responses(
        (status = 200, description = "Conversation deleted", body = DeletedResponse),
        (status = 404, description = "Conversation not found"),
    ),
    security(("bearer_auth" = [])),
    tag = "conversations",
))]
pub async fn delete_conversation(
    State(state): State<AppState>,
    Path(conversation_id): Path<String>,
    req: Request,
) -> Response {
    let tenant_id = match extract_tenant_id(&req) {
        Ok(id) => id,
        Err(err) => return err,
    };

    match state
        .exec_ctx
        .conv_handler
        .store()
        .delete(&tenant_id, &conversation_id)
        .await
    {
        Ok(()) => {
            let response = DeletedResponse::conversation(conversation_id);
            axum::Json(response).into_response()
        }
        Err(agentic_core::storage::StorageError::NotFound { .. }) => {
            error_response(StatusCode::NOT_FOUND, "not_found", "Conversation not found")
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("Failed to delete conversation: {e}"),
        ),
    }
}

pub(super) fn storage_error(error: StorageError) -> Response {
    executor_error_response(ExecutorError::Storage(error))
}

pub(super) fn conversation_response(data: ConversationData) -> Response {
    let metadata = match data.metadata.as_deref().map(serde_json::from_str).transpose() {
        Ok(metadata) => metadata,
        Err(error) => return storage_error(StorageError::Serialization(error)),
    };
    axum::Json(ConversationResponse::new(
        data.conversation_id,
        data.created_at,
        metadata,
    ))
    .into_response()
}
