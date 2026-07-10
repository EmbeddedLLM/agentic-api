use std::collections::HashMap;
use std::fmt;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use http::header::{HeaderName, HeaderValue};
use rmcp::ClientHandler;
use rmcp::ServiceExt;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientCapabilities, ClientInfo, ClientRequest, Implementation,
    InitializeRequestParams, ProtocolVersion, ReadResourceRequestParams, ReadResourceResult, ServerResult, Tool,
};
use rmcp::service::{ClientInitializeError, PeerRequestOptions, RoleClient, RunningService, ServiceError};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::child_process::TokioChildProcess;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

const CONNECTION_TIMEOUT: Duration = Duration::from_secs(30);
const TOOL_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpOperation {
    Connect,
    ListTools,
    CallTool,
    ReadResource,
}

impl fmt::Display for McpOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connect => f.write_str("connect"),
            Self::ListTools => f.write_str("tools/list"),
            Self::CallTool => f.write_str("tools/call"),
            Self::ReadResource => f.write_str("resources/read"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("failed to spawn MCP stdio server")]
    SpawnStdio(#[source] std::io::Error),

    #[error("failed to connect to MCP server")]
    Connect(#[source] Box<ClientInitializeError>),

    #[error("invalid MCP HTTP header name")]
    InvalidHeaderName(#[source] http::header::InvalidHeaderName),

    #[error("invalid MCP HTTP header value")]
    InvalidHeaderValue(#[source] http::header::InvalidHeaderValue),

    #[error("MCP operation failed during {operation}")]
    Operation {
        operation: McpOperation,
        #[source]
        source: ServiceError,
    },

    #[error("MCP operation timed out during {operation}")]
    Timeout { operation: McpOperation },

    #[error("MCP tool arguments must be a JSON object")]
    InvalidArguments,

    #[error("MCP server returned an unexpected response during {operation}")]
    UnexpectedResponse { operation: McpOperation },
}

#[derive(Clone)]
struct AgenticMcpClientHandler;

impl ClientHandler for AgenticMcpClientHandler {
    fn get_info(&self) -> ClientInfo {
        InitializeRequestParams::new(
            ClientCapabilities::default(),
            Implementation::new("agentic-api", env!("CARGO_PKG_VERSION")),
        )
        .with_protocol_version(ProtocolVersion::V_2025_06_18)
    }
}

pub struct McpClient {
    inner: Arc<RunningService<RoleClient, AgenticMcpClientHandler>>,
    tool_timeout: Duration,
}

impl McpClient {
    /// Connects to an MCP server over streamable HTTP.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::Connect`] if the MCP initialization handshake fails.
    pub async fn connect(server_url: &str, headers: Option<HashMap<String, String>>) -> Result<Self, McpError> {
        let mut config = StreamableHttpClientTransportConfig::with_uri(server_url.to_owned());
        if let Some(headers) = headers.filter(|headers| !headers.is_empty()) {
            let mut custom_headers = HashMap::with_capacity(headers.len());
            for (name, value) in headers {
                custom_headers.insert(
                    HeaderName::try_from(name).map_err(McpError::InvalidHeaderName)?,
                    HeaderValue::try_from(value).map_err(McpError::InvalidHeaderValue)?,
                );
            }
            config = config.custom_headers(custom_headers);
        }
        let transport = StreamableHttpClientTransport::from_config(config);
        let service = tokio::time::timeout(CONNECTION_TIMEOUT, AgenticMcpClientHandler.serve(transport))
            .await
            .map_err(|_| McpError::Timeout {
                operation: McpOperation::Connect,
            })?
            .map_err(|error| McpError::Connect(Box::new(error)))?;

        Ok(Self {
            inner: Arc::new(service),
            tool_timeout: TOOL_TIMEOUT,
        })
    }

    /// Spawns a local stdio MCP server and connects over stdin/stdout.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::SpawnStdio`] if the process cannot be spawned.
    /// Returns [`McpError::Connect`] if the MCP initialization handshake fails.
    pub async fn connect_stdio(
        command: &str,
        args: &[String],
        env: Option<&HashMap<String, String>>,
        cwd: Option<&str>,
    ) -> Result<Self, McpError> {
        let mut command_builder = Command::new(command);
        command_builder
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .args(args);

        if let Some(env) = env {
            command_builder.envs(env);
        }

        if let Some(cwd) = cwd {
            command_builder.current_dir(cwd);
        }

        let (transport, stderr) = TokioChildProcess::builder(command_builder)
            .spawn()
            .map_err(McpError::SpawnStdio)?;

        if let Some(stderr) = stderr {
            let command = command.to_owned();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr).lines();
                loop {
                    match reader.next_line().await {
                        Ok(Some(line)) => {
                            tracing::info!(mcp.command = %command, %line, "MCP server stderr");
                        }
                        Ok(None) => break,
                        Err(error) => {
                            tracing::warn!(
                                mcp.command = %command,
                                error = %error,
                                "failed to read MCP server stderr"
                            );
                            break;
                        }
                    }
                }
            });
        }

        let service = tokio::time::timeout(CONNECTION_TIMEOUT, AgenticMcpClientHandler.serve(transport))
            .await
            .map_err(|_| McpError::Timeout {
                operation: McpOperation::Connect,
            })?
            .map_err(|error| McpError::Connect(Box::new(error)))?;

        Ok(Self {
            inner: Arc::new(service),
            tool_timeout: TOOL_TIMEOUT,
        })
    }

    /// Lists tools exposed by the connected MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::Timeout`] if `tools/list` exceeds the configured timeout.
    /// Returns [`McpError::Operation`] if the server rejects or fails the request.
    pub async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        let result = tokio::time::timeout(self.tool_timeout, self.inner.list_tools(None))
            .await
            .map_err(|_| McpError::Timeout {
                operation: McpOperation::ListTools,
            })?
            .map_err(|source| McpError::Operation {
                operation: McpOperation::ListTools,
                source,
            })?;

        Ok(result.tools)
    }

    /// Calls a tool exposed by the connected MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::InvalidArguments`] if `arguments` is not a JSON object.
    /// Returns [`McpError::Timeout`] if `tools/call` exceeds the configured timeout.
    /// Returns [`McpError::Operation`] if the server rejects or fails the request.
    /// Returns [`McpError::UnexpectedResponse`] if the server returns another response kind.
    pub async fn call_tool(&self, name: &str, arguments: Option<Value>) -> Result<CallToolResult, McpError> {
        let arguments = match arguments {
            Some(Value::Object(map)) => Some(map),
            Some(_) => return Err(McpError::InvalidArguments),
            None => None,
        };

        let mut params = CallToolRequestParams::new(name.to_owned());
        params.arguments = arguments;

        let result = tokio::time::timeout(self.tool_timeout, async {
            self.inner
                .peer()
                .send_request_with_option(
                    ClientRequest::CallToolRequest(rmcp::model::CallToolRequest::new(params)),
                    PeerRequestOptions::no_options(),
                )
                .await?
                .await_response()
                .await
        })
        .await
        .map_err(|_| McpError::Timeout {
            operation: McpOperation::CallTool,
        })?
        .map_err(|source| McpError::Operation {
            operation: McpOperation::CallTool,
            source,
        })?;

        match result {
            ServerResult::CallToolResult(result) => Ok(result),
            _ => Err(McpError::UnexpectedResponse {
                operation: McpOperation::CallTool,
            }),
        }
    }

    /// Reads a resource by URI from the connected MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::Timeout`] if `resources/read` exceeds the configured timeout.
    /// Returns [`McpError::Operation`] if the server rejects or fails the request.
    pub async fn read_resource(&self, uri: &str) -> Result<ReadResourceResult, McpError> {
        tokio::time::timeout(
            self.tool_timeout,
            self.inner.read_resource(ReadResourceRequestParams::new(uri.to_owned())),
        )
        .await
        .map_err(|_| McpError::Timeout {
            operation: McpOperation::ReadResource,
        })?
        .map_err(|source| McpError::Operation {
            operation: McpOperation::ReadResource,
            source,
        })
    }
}
