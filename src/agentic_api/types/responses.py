"""Minimal OpenAI Responses API types for agentic-api.

Covers only what the rehydration store and engine need:
- Request / response shapes
- Input and output item types (text messages + function calls only — no tools)
- AgentRunSettings passed to pydantic_ai
"""

from typing import Any, Literal, TypedDict, Self

from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings

from agentic_api.utils.common import uuid7_str

# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


class InputTextContent(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


class InputImageContent(BaseModel):
    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    detail: str | None = None


InputContent = InputTextContent | InputImageContent


class InputMessage(BaseModel):
    """A single user/system/assistant message in the input list."""

    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system", "developer"]
    content: str | list[InputContent]


class FunctionToolResultMessage(BaseModel):
    """Result of a function (user-defined) tool call, fed back into the conversation."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


# Union of all valid input item types.
InputItem = InputMessage | FunctionToolResultMessage


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


class OutputTextContent(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[Any] = Field(default_factory=list)


class OutputMessage(BaseModel):
    type: Literal["message"] = "message"
    id: str
    role: Literal["assistant"] = "assistant"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    content: list[OutputTextContent] = Field(default_factory=list)


class FunctionToolCall(BaseModel):
    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str
    status: Literal["in_progress", "completed"] = "completed"


# Union of all valid output item types.
OutputItem = OutputMessage | FunctionToolCall


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class InputTokenDetails(BaseModel):
    cached_tokens: int = 0


class OutputTokenDetails(BaseModel):
    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: InputTokenDetails = Field(default_factory=InputTokenDetails)
    output_tokens_details: OutputTokenDetails = Field(
        default_factory=OutputTokenDetails
    )


# ---------------------------------------------------------------------------
# Tool types (user-defined function tools only — no built-ins)
# ---------------------------------------------------------------------------


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


ResponsesTool = FunctionTool


class AutoToolChoice(BaseModel):
    type: Literal["auto"] = "auto"


class NoneToolChoice(BaseModel):
    type: Literal["none"] = "none"


class RequiredToolChoice(BaseModel):
    type: Literal["required"] = "required"


class FunctionToolChoice(BaseModel):
    type: Literal["function"] = "function"
    name: str


ToolChoice = AutoToolChoice | NoneToolChoice | RequiredToolChoice | FunctionToolChoice


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------


class ResponsesRequest(BaseModel):
    """Inbound request to POST /v1/responses."""

    model: str
    input: str | list[InputItem]
    instructions: str | None = None
    previous_response_id: str | None = None
    conversation_id: str | None = None
    tools: list[ResponsesTool] | None = None
    tool_choice: ToolChoice = Field(default_factory=AutoToolChoice)
    stream: bool = False
    response_store_enabled: bool = True
    # TODO: remove when conversation handling gets its own router; control moves there
    conversation_store_enabled: bool = False
    include: list[str] | None = None
    # Pass-through fields forwarded verbatim to the upstream.
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    truncation: str | None = None
    metadata: dict[str, Any] | None = None

    def as_openai_chat_settings(self) -> ModelSettings:
        settings: dict[str, Any] = {}
        if self.temperature is not None:
            settings["temperature"] = self.temperature
        if self.top_p is not None:
            settings["top_p"] = self.top_p
        if self.max_output_tokens is not None:
            settings["max_tokens"] = self.max_output_tokens
        return ModelSettings(**settings)


class IncompleteDetails(BaseModel):
    reason: str | None = None


class ResponsesResponse(BaseModel):
    """Completed response returned from the engine or loaded from the store."""

    id: str
    object: Literal["response"] = "response"
    created_at: int = 0
    model: str = ""
    status: Literal["in_progress", "completed", "incomplete", "failed"] = "in_progress"
    output: list[OutputItem] = Field(default_factory=list)
    usage: ResponseUsage | None = None
    incomplete_details: IncompleteDetails | None = None
    error: dict[str, Any] | None = None
    previous_response_id: str | None = None
    conversation_id: str | None = None
    instructions: str | None = None

    def as_responses_chunk(self) -> str:
        return f"data: {self.model_dump_json()}\n\n"

    @classmethod
    def create_from_response_request(cls, request: ResponsesRequest) -> Self:
        return ResponsesResponse(
            id=uuid7_str("resp_"),
            model=request.model,
            instructions=request.instructions,
            previous_response_id=request.previous_response_id,
        )


# ---------------------------------------------------------------------------
# Stream event wrappers (used by composer / SSE layer)
# ---------------------------------------------------------------------------


class ResponseEvent(BaseModel):
    """Wrapper for a lifecycle event in the SSE stream."""

    type: str
    sequence_number: int
    response: ResponsesResponse

    def as_responses_chunk(self) -> str:
        return f"event: {self.type}\ndata: {self.model_dump_json()}\n\n"


class OutputItemEvent(BaseModel):
    """Wrapper for an output item added/done event."""

    type: str
    sequence_number: int
    output_index: int
    item: OutputItem

    def as_responses_chunk(self) -> str:
        return f"event: {self.type}\ndata: {self.model_dump_json()}\n\n"


class ContentPartEvent(BaseModel):
    """Wrapper for a content part added/done event."""

    type: str
    sequence_number: int
    output_index: int
    content_index: int
    item_id: str
    part: OutputTextContent | None = None

    def as_responses_chunk(self) -> str:
        return f"event: {self.type}\ndata: {self.model_dump_json()}\n\n"


class TextDeltaEvent(BaseModel):
    """Wrapper for text/argument delta and done events."""

    type: str
    sequence_number: int
    output_index: int
    item_id: str
    content_index: int = 0
    delta: str | None = None
    text: str | None = None
    arguments: str | None = None
    logprobs: list[Any] = Field(default_factory=list)

    def as_responses_chunk(self) -> str:
        return f"event: {self.type}\ndata: {self.model_dump_json()}\n\n"


class ErrorEvent(BaseModel):
    type: Literal["response.error"] = "response.error"
    sequence_number: int
    code: str
    message: str
    param: str | None = None

    def as_responses_chunk(self) -> str:
        return f"event: {self.type}\ndata: {self.model_dump_json()}\n\n"


# Union used by the composer and SSE layer.
StreamEvent = (
    ResponseEvent | OutputItemEvent | ContentPartEvent | TextDeltaEvent | ErrorEvent
)


# ---------------------------------------------------------------------------
# pydantic_ai run settings
# ---------------------------------------------------------------------------


class AgentRunSettings(TypedDict):
    message_history: list[ModelMessage]
    instructions: str | None
    toolsets: list[Any]
    usage_limits: Any | None
