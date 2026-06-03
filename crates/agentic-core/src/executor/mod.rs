//! Agentic loop executor.

pub mod accumulator;
pub mod engine;
pub mod error;
pub mod modes;
pub mod request;

pub use engine::{BoxStream, call_inference, create_conversation, execute, persist_response, rehydrate_conversation};
pub use error::{ExecutorError, ExecutorResult};
pub use modes::{ConversationHandler, ResponseHandler};
pub use request::ExecutionContext;
pub use request::RequestContext;
