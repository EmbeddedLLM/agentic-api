//! Agentic loop executor.

pub mod accumulator;
pub mod dispatch;
pub mod engine;
pub mod error;
pub mod modes;
pub mod request;

pub use dispatch::{LoopDecision, client_action_items, decide_client_action};
pub use engine::{
    BoxStream, call_inference, create_conversation, execute, persist_response, prepare_context_for_upstream,
    rehydrate_conversation, upstream_request_json,
};
pub use error::{ExecutorError, ExecutorResult};
pub use modes::{ConversationHandler, ResponseHandler};
pub use request::ExecutionContext;
pub use request::RequestContext;
