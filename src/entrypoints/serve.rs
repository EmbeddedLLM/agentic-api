use std::time::Duration;

use tokio::net::TcpListener;
use tracing::info;

use crate::config::RuntimeConfig;
use crate::entrypoints::app::build_router;
use crate::entrypoints::proxy::ProxyState;

/// Poll upstream `/health` until it responds 200 or the timeout is reached.
///
/// # Errors
///
/// Returns an error if the upstream does not become ready within the configured timeout.
pub async fn wait_upstream_ready(config: &RuntimeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let base = config.llm_api_base.trim_end_matches('/');
    let url = format!("{base}/health");

    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = config.openai_api_key.as_deref() {
        let trimmed = key.trim();
        if !trimmed.is_empty() {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&format!("Bearer {trimmed}"))?,
            );
        }
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .default_headers(headers)
        .build()?;

    let timeout = Duration::from_secs_f64(config.upstream_ready_timeout_s);
    let interval = Duration::from_secs_f64(config.upstream_ready_interval_s);
    let start = tokio::time::Instant::now();
    let mut last_notice = Duration::ZERO;

    loop {
        let elapsed = start.elapsed();
        if elapsed > timeout {
            return Err(format!(
                "vLLM did not become ready within {:.0}s: {url}",
                config.upstream_ready_timeout_s
            )
            .into());
        }

        match client.get(&url).send().await {
            Ok(resp) if resp.status().as_u16() == 200 => return Ok(()),
            _ => {}
        }

        if elapsed.saturating_sub(last_notice) >= interval {
            last_notice = elapsed;
            info!("waiting for upstream ({}s elapsed): {url}", elapsed.as_secs());
        }

        tokio::time::sleep(interval).await;
    }
}

/// Start the gateway after the upstream becomes ready.
///
/// # Errors
///
/// Returns an error if upstream readiness polling fails or the server cannot bind.
pub async fn run(config: RuntimeConfig) -> Result<(), Box<dyn std::error::Error>> {
    wait_upstream_ready(&config).await?;
    info!("upstream ready: {}", config.llm_api_base);

    let addr = format!("{}:{}", config.gateway_host, config.gateway_port);
    let state = ProxyState::new(config);
    let router = build_router(state);
    let listener = TcpListener::bind(&addr).await?;
    info!("gateway listening on {addr}");
    axum::serve(listener, router).await?;
    Ok(())
}

/// Spawn vLLM as a subprocess and run the gateway in the foreground.
///
/// # Errors
///
/// Returns an error if vLLM fails to start or the gateway errors.
pub async fn run_with_vllm(config: RuntimeConfig, vllm_args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = tokio::process::Command::new("python");
    cmd.arg("-m").arg("vllm.entrypoints.openai.api_server");
    cmd.args(&vllm_args);

    let mut child = cmd.spawn()?;
    info!("spawned vLLM subprocess (pid {})", child.id().unwrap_or(0));

    let result = run(config).await;

    let _ = child.kill().await;
    let _ = child.wait().await;
    result
}
