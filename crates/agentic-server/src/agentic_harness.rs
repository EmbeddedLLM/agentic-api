use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub struct HarnessEnv {
    pub environment: BTreeMap<String, String>,
    pub files: Vec<PathBuf>,
    pub summary: String,
}

/// Write an isolated Codex home for an Agentic API session.
///
/// # Errors
///
/// Returns an I/O error when the temporary home or generated files cannot be written.
pub fn prepare_codex_home(
    root: &Path,
    gateway_url: &str,
    model: &str,
    api_key: Option<&str>,
) -> Result<HarnessEnv, io::Error> {
    fs::create_dir_all(root)?;
    let gateway_url = format!("{}/v1", gateway_url.trim_end_matches('/'));
    let catalog_path = root.join("model_catalog.json");
    let config_path = root.join("config.toml");
    let catalog = serde_json::json!({
        "models": [{
            "slug": model,
            "display_name": model,
            "supported_in_api": true,
            "visibility": "list",
            "input_modalities": ["text"],
            "supported_reasoning_levels": [
                {"effort": "low", "description": "Fast responses"},
                {"effort": "medium", "description": "Balanced responses"},
                {"effort": "high", "description": "Deep reasoning"}
            ],
            "supports_reasoning_summaries": true,
            "supports_parallel_tool_calls": true,
            "apply_patch_tool_type": "freeform",
            "web_search_tool_type": "text"
        }]
    });
    let catalog_bytes = serde_json::to_vec_pretty(&catalog).map_err(io::Error::other)?;
    fs::write(&catalog_path, catalog_bytes).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to write Codex model catalog {}: {error}",
                catalog_path.display()
            ),
        )
    })?;

    let requires_auth = api_key.is_some();
    let config = format!(
        "model = \"{}\"\\nmodel_provider = \"agentic-api\"\\nmodel_catalog_json = \"{}\"\\n\\n\
[model_providers.agentic-api]\\nname = \"Agentic API\"\\nbase_url = \"{}\"\\n\
wire_api = \"responses\"\\nrequires_openai_auth = {requires_auth}\\nsupports_websockets = true\\n",
        toml_escape(model),
        toml_escape(&catalog_path.display().to_string()),
        toml_escape(&gateway_url),
    );
    fs::write(&config_path, config).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to write Codex config {}: {error}", config_path.display()),
        )
    })?;

    let mut environment = BTreeMap::new();
    environment.insert("CODEX_HOME".to_owned(), root.display().to_string());
    if let Some(api_key) = api_key {
        environment.insert("OPENAI_API_KEY".to_owned(), api_key.to_owned());
    }

    Ok(HarnessEnv {
        environment,
        files: vec![config_path, catalog_path],
        summary: format!("Codex home: {} (model: {model})", root.display()),
    })
}

#[must_use]
pub fn prepare_claude_env(gateway_url: &str, model: &str, api_key: Option<&str>) -> HarnessEnv {
    let mut environment = BTreeMap::from([
        (
            "ANTHROPIC_BASE_URL".to_owned(),
            gateway_url.trim_end_matches('/').to_owned(),
        ),
        ("ANTHROPIC_MODEL".to_owned(), model.to_owned()),
        ("ANTHROPIC_SMALL_FAST_MODEL".to_owned(), model.to_owned()),
        (
            "ANTHROPIC_API_KEY".to_owned(),
            api_key.unwrap_or("agentic-api-local").to_owned(),
        ),
    ]);
    if let Some(api_key) = api_key {
        environment.insert("OPENAI_API_KEY".to_owned(), api_key.to_owned());
    }
    HarnessEnv {
        environment,
        files: Vec::new(),
        summary: format!(
            "Claude Code gateway: {} (model: {model})",
            gateway_url.trim_end_matches('/')
        ),
    }
}

fn toml_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\"', "\\\"")
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{prepare_claude_env, prepare_codex_home};

    #[test]
    fn codex_config_is_isolated_and_contains_gateway_provider() {
        let root = unique_temp_dir("codex");
        let env = prepare_codex_home(&root, "http://127.0.0.1:3000", "Qwen/test", None).expect("config");
        let config = fs::read_to_string(root.join("config.toml")).expect("config file");

        assert!(config.contains("[model_providers.agentic-api]"));
        assert!(config.contains("base_url = \"http://127.0.0.1:3000/v1\""));
        assert!(config.contains("wire_api = \"responses\""));
        assert!(config.contains("requires_openai_auth = false"));
        assert!(config.contains("model = \"Qwen/test\""));
        assert!(!env.summary.contains("secret"));

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn claude_environment_uses_gateway_and_does_not_expose_key() {
        let env = prepare_claude_env("http://127.0.0.1:3000", "Qwen/test", Some("secret-key"));

        assert_eq!(
            env.environment.get("ANTHROPIC_BASE_URL"),
            Some(&"http://127.0.0.1:3000".to_owned())
        );
        assert_eq!(env.environment.get("ANTHROPIC_MODEL"), Some(&"Qwen/test".to_owned()));
        assert_eq!(env.environment.get("ANTHROPIC_API_KEY"), Some(&"secret-key".to_owned()));
        assert!(!env.summary.contains("secret-key"));
    }

    fn unique_temp_dir(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("agentic-api-{name}-{}", std::process::id()));
        fs::create_dir_all(&path).expect("temp dir");
        path
    }
}
