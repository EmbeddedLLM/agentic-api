DROP INDEX IF EXISTS idx_items_conversation_id;
CREATE UNIQUE INDEX idx_items_conversation_id ON items (conversation_id, seq);
