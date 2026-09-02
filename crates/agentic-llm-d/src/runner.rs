//! Build state, bind, serve, drain.

use std::future::IntoFuture;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use agentic_core::config::Config;
use agentic_core::executor::ExecutionContext;

use crate::{BackendState, build_router};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to build the execution context: {0}")]
    Context(#[from] agentic_core::error::Error),
    #[error("failed to bind or serve: {0}")]
    Io(#[from] std::io::Error),
}

/// Matches the gateway's bound: a drain that never finishes would outlive the
/// pod's termination grace period and be killed mid-request anyway.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(8);

/// Opens storage from `config` and serves the backend router until `shutdown`,
/// then drains in-flight requests for at most [`DRAIN_TIMEOUT`].
///
/// # Errors
/// If the storage pool cannot be opened, or the address cannot be bound.
pub async fn serve(
    config: &Config,
    signing_key: Vec<u8>,
    api_token: String,
    host: &str,
    port: u16,
    shutdown: CancellationToken,
) -> Result<(), Error> {
    let exec_ctx = Arc::new(ExecutionContext::from_config(config).await?);
    let signing_key = Arc::new(signing_key);
    let api_token = Arc::new(api_token);
    let listener = TcpListener::bind(format!("{host}:{port}")).await?;
    info!("agentic-llm-d listening on {host}:{port} — no proxy, no inference");
    let graceful = shutdown.clone();
    let serving = axum::serve(
        listener,
        build_router(BackendState {
            exec_ctx,
            signing_key,
            api_token,
        }),
    )
    .with_graceful_shutdown(async move { graceful.cancelled().await })
    .into_future();
    tokio::pin!(serving);

    tokio::select! {
        result = &mut serving => result?,
        () = shutdown.cancelled() => {
            if tokio::time::timeout(DRAIN_TIMEOUT, serving).await.is_err() {
                warn!(
                    timeout_seconds = DRAIN_TIMEOUT.as_secs(),
                    "drain timed out; closing remaining connections"
                );
            }
        }
    }
    Ok(())
}
