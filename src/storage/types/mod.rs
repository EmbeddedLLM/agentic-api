//! Domain types for storage operations.

pub mod conversation;
pub mod errors;
pub mod item;
pub mod response;

pub use conversation::ConversationData;
pub use errors::{Result, StorageError};
pub use item::{InOutItem, ItemKind};
pub use response::{ResponseData, ResponseMetadata};
