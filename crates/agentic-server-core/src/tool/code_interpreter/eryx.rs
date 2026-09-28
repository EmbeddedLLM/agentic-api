//! Eryx implementation of the code interpreter provider.

use std::future::Future;
use std::num::NonZeroUsize;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use eryx::{CancellationToken, Error as EryxError, OutputHandler, ResourceLimits, Sandbox};

use crate::config::CodeInterpreterRuntimeConfig;
use crate::tool::ToolError;

use super::provider::{CodeInterpreterProvider, ExecutionCancellation, ExecutionOutput, ExecutionStatus};

const TRUNCATION_MARKER: &str = "\n[output truncated]";

#[derive(Debug)]
struct OutputState {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_limit: NonZeroUsize,
    stderr_limit: NonZeroUsize,
    stdout_exceeded: bool,
    stderr_exceeded: bool,
    cancellation: Option<CancellationToken>,
}

impl OutputState {
    fn new(stdout_limit: NonZeroUsize, stderr_limit: NonZeroUsize) -> Self {
        Self {
            stdout: Vec::with_capacity(stdout_limit.get().min(8 * 1024)),
            stderr: Vec::with_capacity(stderr_limit.get().min(8 * 1024)),
            stdout_limit,
            stderr_limit,
            stdout_exceeded: false,
            stderr_exceeded: false,
            cancellation: None,
        }
    }

    fn install_cancellation(&mut self, cancellation: CancellationToken) {
        if self.stdout_exceeded || self.stderr_exceeded {
            cancellation.cancel();
        }
        self.cancellation = Some(cancellation);
    }

    fn append(&mut self, chunk: &[u8], is_stderr: bool) {
        let exceeded = if is_stderr {
            append_bounded(&mut self.stderr, chunk, self.stderr_limit)
        } else {
            append_bounded(&mut self.stdout, chunk, self.stdout_limit)
        };
        if exceeded {
            if is_stderr {
                self.stderr_exceeded = true;
            } else {
                self.stdout_exceeded = true;
            }
            if let Some(cancellation) = &self.cancellation {
                cancellation.cancel();
            }
        }
    }

    fn take_output(&mut self) -> (String, String, bool) {
        add_truncation_marker(&mut self.stdout, self.stdout_limit, self.stdout_exceeded);
        add_truncation_marker(&mut self.stderr, self.stderr_limit, self.stderr_exceeded);
        let exceeded = self.stdout_exceeded || self.stderr_exceeded;
        (
            String::from_utf8_lossy(&std::mem::take(&mut self.stdout)).into_owned(),
            String::from_utf8_lossy(&std::mem::take(&mut self.stderr)).into_owned(),
            exceeded,
        )
    }
}

fn append_bounded(buffer: &mut Vec<u8>, chunk: &[u8], limit: NonZeroUsize) -> bool {
    let remaining = limit.get().saturating_sub(buffer.len());
    buffer.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    chunk.len() > remaining
}

fn add_truncation_marker(buffer: &mut Vec<u8>, limit: NonZeroUsize, exceeded: bool) {
    if !exceeded {
        return;
    }
    let marker = TRUNCATION_MARKER.as_bytes();
    let marker_len = marker.len().min(limit.get());
    buffer.truncate(limit.get().saturating_sub(marker_len));
    buffer.extend_from_slice(&marker[..marker_len]);
}

#[derive(Clone)]
struct BoundedOutputHandler(Arc<Mutex<OutputState>>);

#[async_trait]
impl OutputHandler for BoundedOutputHandler {
    async fn on_output(&self, chunk: &[u8]) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .append(chunk, false);
    }

    async fn on_stderr(&self, chunk: &[u8]) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .append(chunk, true);
    }
}

#[derive(Debug)]
pub(super) struct EryxProvider {
    config: CodeInterpreterRuntimeConfig,
}

impl EryxProvider {
    pub(super) fn new(config: CodeInterpreterRuntimeConfig) -> Result<Self, ToolError> {
        ensure_private_temp_directory(std::env::temp_dir().as_path())?;
        Ok(Self { config })
    }
}

impl CodeInterpreterProvider for EryxProvider {
    fn check_ready(&self) -> Result<(), ToolError> {
        // Fail startup before registration if the operator did not provide
        // an Eryx 0.8-compatible precompiled runtime.
        build_sandbox(self.config, None)
            .map(|_| ())
            .map_err(|error| map_initialization_error(&error))
    }

    fn max_concurrency(&self) -> NonZeroUsize {
        self.config.max_concurrent_guests
    }

    fn execute(
        &self,
        code: String,
        cancellation: Arc<ExecutionCancellation>,
    ) -> Pin<Box<dyn Future<Output = Result<ExecutionOutput, ToolError>> + Send + '_>> {
        let config = self.config;
        Box::pin(async move { supervise(config, code, cancellation).await })
    }
}

async fn supervise(
    config: CodeInterpreterRuntimeConfig,
    code: String,
    cancellation: Arc<ExecutionCancellation>,
) -> Result<ExecutionOutput, ToolError> {
    let output_state = Arc::new(Mutex::new(OutputState::new(
        config.max_stdout_bytes,
        config.max_stderr_bytes,
    )));
    let handler = BoundedOutputHandler(Arc::clone(&output_state));
    let sandbox = tokio::task::spawn_blocking(move || build_sandbox(config, Some(handler)))
        .await
        .map_err(|_| ToolError::Execution("code interpreter initialization task failed".to_owned()))?
        .map_err(|error| map_initialization_error(&error))?;

    let handle = sandbox.execute_cancellable(&code);
    let handle_cancellation = handle.cancellation_token();
    cancellation.install({
        let handle_cancellation = handle_cancellation.clone();
        move || handle_cancellation.cancel()
    });
    output_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .install_cancellation(handle_cancellation);
    let result = handle.wait().await;
    let (stdout, mut stderr, output_exceeded) = output_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take_output();

    let status = match result {
        // Crossing an output budget cancels the guest, so a nominal success
        // and the resulting cancellation both mean the run was cut short.
        Ok(_) | Err(EryxError::Cancelled) if output_exceeded => ExecutionStatus::Incomplete,
        Ok(_) => ExecutionStatus::Completed,
        Err(EryxError::PythonException(_)) => {
            if stderr.is_empty() {
                "Python execution raised an exception.".clone_into(&mut stderr);
            }
            ExecutionStatus::Failed
        }
        Err(EryxError::Timeout(_) | EryxError::FuelExhausted { .. } | EryxError::ResourceLimit(_)) => {
            if stderr.is_empty() {
                "Code execution exceeded a resource limit.".clone_into(&mut stderr);
            }
            ExecutionStatus::Incomplete
        }
        Err(_) => return Err(ToolError::Execution("code interpreter runtime failed".to_owned())),
    };
    Ok(ExecutionOutput { status, stdout, stderr })
}

fn build_sandbox(
    config: CodeInterpreterRuntimeConfig,
    output_handler: Option<BoundedOutputHandler>,
) -> Result<Sandbox, EryxError> {
    let memory = u64::try_from(config.max_guest_memory_bytes.get())
        .map_err(|_| EryxError::Initialization("guest memory limit is too large".to_owned()))?;
    let limits = ResourceLimits::default()
        .with_execution_timeout(config.execution_wall_time)
        .with_max_memory_bytes(memory)
        .with_max_fuel(config.max_fuel.get());
    let builder = Sandbox::embedded()
        .with_trace_collection(false)
        .with_resource_limits(limits)
        .with_result_variable(format!("__agentic_private_result_{}", uuid::Uuid::now_v7().simple()));
    match output_handler {
        Some(handler) => builder.with_output_handler(handler).build(),
        None => builder.build(),
    }
}

fn ensure_private_temp_directory(temp_dir: &Path) -> Result<(), ToolError> {
    std::fs::create_dir_all(temp_dir)
        .map_err(|_| ToolError::Config("code interpreter TMPDIR is not writable".to_owned()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mode = std::fs::metadata(temp_dir)
            .map_err(|_| ToolError::Config("code interpreter TMPDIR cannot be inspected".to_owned()))?
            .permissions()
            .mode()
            & 0o777;
        if mode != 0o700 {
            return Err(ToolError::Config(
                "code interpreter requires an operator-owned TMPDIR with mode 0700".to_owned(),
            ));
        }
    }
    Ok(())
}

/// Record the operator-actionable cause and return a fixed public message.
///
/// Eryx initialization errors can embed host paths from `$TMPDIR` or the
/// embedded-asset cache, so the mapped error stays fixed for callers while
/// the preserved source is written only to the server log.
fn map_initialization_error(error: &EryxError) -> ToolError {
    tracing::error!(error = %error, "code interpreter embedded runtime failed to initialize");
    ToolError::Config("code interpreter embedded runtime failed to initialize".to_owned())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::time::Duration;

    use super::*;
    use crate::tool::code_interpreter::CodeInterpreterExecutor;

    fn test_config() -> CodeInterpreterRuntimeConfig {
        CodeInterpreterRuntimeConfig {
            enabled: true,
            execution_wall_time: Duration::from_secs(10),
            max_fuel: NonZeroU64::new(10_000_000_000).expect("nonzero"),
            max_stdout_bytes: NonZeroUsize::new(128).expect("nonzero"),
            max_stderr_bytes: NonZeroUsize::new(128).expect("nonzero"),
            max_concurrent_guests: NonZeroUsize::new(1).expect("nonzero"),
            max_aggregate_guest_memory_bytes: NonZeroUsize::new(128 * 1024 * 1024).expect("nonzero"),
            ..CodeInterpreterRuntimeConfig::default()
        }
    }

    #[tokio::test]
    async fn embedded_runtime_executes_python_and_classifies_failures_and_limits() {
        let executor = CodeInterpreterExecutor::from_config(test_config()).expect("embedded runtime starts");

        let success = executor
            .execute_call(r#"{"code":"print(6 * 7)"}"#)
            .await
            .expect("successful execution");
        assert!(matches!(success.status, ExecutionStatus::Completed));
        assert_eq!(success.stdout.trim(), "42");

        let exception = executor
            .execute_call(r#"{"code":"raise ValueError('private detail')"}"#)
            .await
            .expect("script failure is a typed tool output");
        assert!(matches!(exception.status, ExecutionStatus::Failed));
        assert!(!exception.stderr.contains("private detail"));

        let filesystem = executor
        .execute_call(
            r#"{"code":"import json\nprint('stdlib-imported')\ntry:\n    with open('/data/agentic-write-probe', 'w', encoding='utf-8') as handle:\n        handle.write('probe')\n    print('data-writable')\nexcept OSError:\n    print('data-unavailable')\ntry:\n    with open(json.__file__, 'a', encoding='utf-8') as handle:\n        handle.write('probe')\n    print('stdlib-writable')\nexcept (OSError, TypeError):\n    print('stdlib-read-only')"}"#,
        )
        .await
        .expect("filesystem policy is a typed tool output");
        assert!(matches!(filesystem.status, ExecutionStatus::Completed));
        assert!(filesystem.stdout.contains("stdlib-imported"));
        assert!(filesystem.stdout.contains("data-unavailable"));
        assert!(filesystem.stdout.contains("stdlib-read-only"));
        assert!(!filesystem.stdout.contains("-writable"));

        let oversized = executor
            .execute_call(r#"{"code":"print('x' * 10000)"}"#)
            .await
            .expect("output limit is a typed tool output");
        assert!(matches!(oversized.status, ExecutionStatus::Incomplete));
        assert!(oversized.stdout.len() <= test_config().max_stdout_bytes.get());
        assert!(oversized.stdout.contains("[output truncated]"));
    }
}
