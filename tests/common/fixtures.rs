use agentic_api::types::responses::{
    OutputItem, OutputMessage, OutputTextContent, ResponsesInput, ResponsesRequest, ResponsesResponse, ToolChoice,
};

pub fn make_request(model: &str) -> ResponsesRequest {
    ResponsesRequest {
        model: model.to_string(),
        input: ResponsesInput::Text("hello".to_string()),
        instructions: None,
        previous_response_id: None,
        conversation_id: None,
        tools: None,
        tool_choice: ToolChoice::Auto,
        stream: false,
        response_store_enabled: true,
        conversation_store_enabled: false,
        include: None,
        temperature: None,
        top_p: None,
        max_output_tokens: None,
        truncation: None,
        metadata: None,
    }
}

pub fn make_response(id: &str, model: &str, status: &str) -> ResponsesResponse {
    ResponsesResponse {
        id: id.to_string(),
        object: "response".to_string(),
        created_at: 0,
        model: model.to_string(),
        status: status.to_string(),
        output: vec![OutputItem::Message(OutputMessage {
            id: "msg_1".to_string(),
            role: "assistant".to_string(),
            status: "completed".to_string(),
            content: vec![OutputTextContent::new("hello back")],
        })],
        usage: None,
        incomplete_details: None,
        error: None,
        previous_response_id: None,
        conversation_id: None,
        instructions: None,
    }
}
