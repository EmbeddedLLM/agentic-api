use std::{ffi::OsString, path::Path, time::Duration};

use agentic_core::error::Error;
use reqwest::Client;
use tokio::time::{Instant, sleep};

use crate::agentic_cli::{CommonOptions, SourceOptions};

#[must_use]
pub fn server_args(source: &SourceOptions, common: &CommonOptions) -> Vec<OsString> {
    let mut args = Vec::new();
    if let Some(upstream) = &source.upstream {
        args.extend([OsString::from("--llm-api-base"), OsString::from(upstream)]);
    } else if let Some(model) = &source.model {
        args.extend([OsString::from("serve"), OsString::from(model)]);
        args.extend([OsString::from("--port"), OsString::from(source.llm_port.to_string())]);
    }
    args.extend([
        OsString::from("--gateway-host"),
        OsString::from(&common.gateway_host),
        OsString::from("--gateway-port"),
        OsString::from(common.gateway_port.to_string()),
        OsString::from("--db-url"),
        OsString::from(&common.database_url),
        OsString::from("--llm-ready-timeout-s"),
        OsString::from(common.llm_ready_timeout_s.to_string()),
        OsString::from("--llm-ready-interval-s"),
        OsString::from(common.llm_ready_interval_s.to_string()),
    ]);
    if let Some(api_key) = &common.api_key {
        args.extend([OsString::from("--openai-api-key"), OsString::from(api_key)]);
    }
    if common.skip_llm_ready_check {
        args.push(OsString::from("--skip-llm-ready-check"));
    }
    args
}

#[must_use]
pub fn server_binary_path(current_exe: &Path) -> std::path::PathBuf {
    current_exe.with_file_name("agentic-server")
}

fn harness_launch_args(harness: crate::agentic_cli::Harness, yolo: bool, passthrough: &[String]) -> Vec<String> {
    let mut args = Vec::with_capacity(passthrough.len() + 3);
    if yolo {
        match harness {
            crate::agentic_cli::Harness::Codex => {
                args.push("--dangerously-bypass-approvals-and-sandbox".to_owned());
            }
            crate::agentic_cli::Harness::Claude => {
                args.extend([
                    "--dangerously-skip-permissions".to_owned(),
                    "--effort".to_owned(),
                    "medium".to_owned(),
                ]);
            }
        }
    }
    args.extend_from_slice(passthrough);
    args
}

/// Wait until the gateway is live and, unless skipped, its upstream is ready.
///
/// # Errors
///
/// Returns a configuration error when the timeout expires.
pub async fn wait_for_gateway(
    client: &Client,
    gateway_url: &str,
    timeout: Duration,
    interval: Duration,
    skip_llm_ready_check: bool,
) -> Result<(), Error> {
    let deadline = Instant::now() + timeout;
    let health_url = format!("{}/health", gateway_url.trim_end_matches('/'));
    let ready_url = format!("{}/ready", gateway_url.trim_end_matches('/'));
    loop {
        if Instant::now() >= deadline {
            return Err(Error::Config(format!("gateway did not become ready at {gateway_url}")));
        }
        let health_ok = client
            .get(&health_url)
            .send()
            .await
            .is_ok_and(|response| response.status().is_success());
        let ready_ok = skip_llm_ready_check
            || client
                .get(&ready_url)
                .send()
                .await
                .is_ok_and(|response| response.status().is_success());
        if health_ok && ready_ok {
            return Ok(());
        }
        sleep(interval).await;
    }
}

/// Run one gateway-plus-harness session and return the harness exit status.
///
/// # Errors
///
/// Returns an error when a child cannot start or readiness fails.
pub async fn run_session(
    current_exe: &Path,
    harness: crate::agentic_cli::Harness,
    options: crate::agentic_cli::HarnessOptions,
) -> Result<std::process::ExitStatus, Error> {
    let gateway_url = format!("http://{}:{}", options.common.gateway_host, options.common.gateway_port);
    let session_root = std::env::temp_dir().join(format!(
        "agentic-api-session-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos())
    ));
    tokio::fs::create_dir_all(&session_root).await?;

    let mut server = start_server(current_exe, &options)?;

    let client = match Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(Error::HttpClient)
    {
        Ok(client) => client,
        Err(error) => {
            cleanup(&mut server, &session_root).await;
            return Err(error);
        }
    };
    if let Err(error) = wait_for_gateway(
        &client,
        &gateway_url,
        Duration::from_secs_f64(options.common.llm_ready_timeout_s),
        Duration::from_secs_f64(options.common.llm_ready_interval_s),
        options.common.skip_llm_ready_check,
    )
    .await
    {
        cleanup(&mut server, &session_root).await;
        return Err(error);
    }

    let harness_env = match harness_environment(harness, &gateway_url, &options, &session_root) {
        Ok(environment) => environment,
        Err(error) => {
            cleanup(&mut server, &session_root).await;
            return Err(error);
        }
    };

    let mut harness_child = match spawn_harness(harness, &options, &harness_env) {
        Ok(child) => child,
        Err(error) => {
            cleanup(&mut server, &session_root).await;
            return Err(error);
        }
    };

    let harness_status = tokio::select! {
        status = harness_child.wait() => status?,
        signal = tokio::signal::ctrl_c() => {
            signal?;
            let _ = harness_child.kill().await;
            harness_child.wait().await?
        }
    };
    cleanup(&mut server, &session_root).await;
    Ok(harness_status)
}

fn start_server(
    current_exe: &Path,
    options: &crate::agentic_cli::HarnessOptions,
) -> Result<tokio::process::Child, Error> {
    let server_path = server_binary_path(current_exe);
    if !server_path.is_file() {
        return Err(Error::Config(format!(
            "agentic-server binary not found beside {}; run cargo build -p agentic-server --bins first",
            current_exe.display()
        )));
    }
    let mut server = tokio::process::Command::new(server_path);
    server.args(server_args(&options.source, &options.common));
    server.stdout(std::process::Stdio::inherit());
    server.stderr(std::process::Stdio::inherit());
    Ok(server.spawn()?)
}

fn harness_environment(
    harness: crate::agentic_cli::Harness,
    gateway_url: &str,
    options: &crate::agentic_cli::HarnessOptions,
    session_root: &Path,
) -> Result<crate::agentic_harness::HarnessEnv, Error> {
    let mut environment = match harness {
        crate::agentic_cli::Harness::Codex => crate::agentic_harness::prepare_codex_home(
            session_root,
            gateway_url,
            options.source.model.as_deref().unwrap_or("agentic-api"),
            options.common.api_key.as_deref(),
        )
        .map_err(Error::from),
        crate::agentic_cli::Harness::Claude => Ok(crate::agentic_harness::prepare_claude_env(
            gateway_url,
            options.source.model.as_deref().unwrap_or("agentic-api"),
            options.common.api_key.as_deref(),
        )),
    }?;
    if options.common.yolo && matches!(harness, crate::agentic_cli::Harness::Claude) {
        environment
            .environment
            .insert("CLAUDE_CODE_EFFORT_LEVEL".to_owned(), "medium".to_owned());
    }
    Ok(environment)
}

fn spawn_harness(
    harness: crate::agentic_cli::Harness,
    options: &crate::agentic_cli::HarnessOptions,
    harness_env: &crate::agentic_harness::HarnessEnv,
) -> Result<tokio::process::Child, Error> {
    let binary_name = match harness {
        crate::agentic_cli::Harness::Codex => "codex",
        crate::agentic_cli::Harness::Claude => "claude",
    };
    let override_name = match harness {
        crate::agentic_cli::Harness::Codex => "AGENTIC_CODEX_BIN",
        crate::agentic_cli::Harness::Claude => "AGENTIC_CLAUDE_BIN",
    };
    let binary = std::env::var_os(override_name).unwrap_or_else(|| binary_name.into());
    let mut harness_command = tokio::process::Command::new(binary);
    harness_command.args(harness_launch_args(harness, options.common.yolo, &options.harness_args));
    harness_command.envs(&harness_env.environment);
    harness_command.stdin(std::process::Stdio::inherit());
    harness_command.stdout(std::process::Stdio::inherit());
    harness_command.stderr(std::process::Stdio::inherit());
    harness_command
        .spawn()
        .map_err(|error| Error::Config(format!("failed to launch {binary_name} ({override_name}): {error}")))
}

async fn cleanup(server: &mut tokio::process::Child, session_root: &Path) {
    let _ = server.kill().await;
    let _ = server.wait().await;
    let _ = tokio::fs::remove_dir_all(session_root).await;
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::{harness_launch_args, server_args};
    use crate::agentic_cli::{CommonOptions, Harness, SourceOptions};

    #[test]
    fn integrated_mode_builds_server_arguments() {
        let args = server_args(
            &SourceOptions {
                upstream: None,
                model: Some("Qwen/test".to_owned()),
                llm_port: 8000,
            },
            &CommonOptions::default(),
        );
        let args: Vec<_> = args.iter().map(OsString::as_os_str).collect();

        assert_eq!(args[0], "serve");
        assert_eq!(args[1], "Qwen/test");
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--db-url", "sqlite://./agentic_api.db"])
        );
    }

    #[test]
    fn standalone_mode_builds_upstream_arguments() {
        let args = server_args(
            &SourceOptions {
                upstream: Some("http://127.0.0.1:8000".to_owned()),
                model: None,
                llm_port: 8000,
            },
            &CommonOptions::default(),
        );
        let args: Vec<_> = args.iter().map(OsString::as_os_str).collect();

        assert_eq!(args[0], "--llm-api-base");
        assert_eq!(args[1], "http://127.0.0.1:8000");
    }

    #[test]
    fn explicit_upstream_wins_when_model_names_the_harness_model() {
        let args = server_args(
            &SourceOptions {
                upstream: Some("http://127.0.0.1:8000".to_owned()),
                model: Some("Qwen/test".to_owned()),
                llm_port: 8000,
            },
            &CommonOptions::default(),
        );
        let args: Vec<_> = args.iter().map(OsString::as_os_str).collect();

        assert_eq!(args[0], "--llm-api-base");
        assert!(!args.iter().any(|arg| *arg == "serve"));
    }

    #[test]
    fn yolo_mode_uses_native_codex_bypass_flag() {
        assert_eq!(
            harness_launch_args(Harness::Codex, true, &["exec".to_owned()]),
            ["--dangerously-bypass-approvals-and-sandbox", "exec"]
        );
    }

    #[test]
    fn yolo_mode_uses_native_claude_bypass_and_compatible_effort() {
        assert_eq!(
            harness_launch_args(Harness::Claude, true, &[]),
            ["--dangerously-skip-permissions", "--effort", "medium"]
        );
    }

    #[test]
    fn yolo_claude_environment_overrides_inherited_effort() {
        let options = crate::agentic_cli::HarnessOptions {
            source: SourceOptions {
                upstream: Some("http://127.0.0.1:8000".to_owned()),
                model: Some("Qwen/test".to_owned()),
                llm_port: 8000,
            },
            common: CommonOptions {
                yolo: true,
                ..CommonOptions::default()
            },
            harness_args: Vec::new(),
        };
        let root = std::env::temp_dir().join(format!("agentic-api-yolo-test-{}", std::process::id()));
        let environment = super::harness_environment(Harness::Claude, "http://127.0.0.1:3000", &options, &root)
            .expect("Claude environment");

        assert_eq!(
            environment.environment.get("CLAUDE_CODE_EFFORT_LEVEL"),
            Some(&"medium".to_owned())
        );
    }
}
