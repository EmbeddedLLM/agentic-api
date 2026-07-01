use std::collections::HashMap;
use std::hash::BuildHasher;

#[derive(Debug, Clone)]
pub struct Config {
    pub llm_api_base: String,
    pub openai_api_key: Option<String>,
    pub llm_ready_timeout_s: f64,
    pub llm_ready_interval_s: f64,
    pub skip_llm_ready_check: bool,
    /// Database URL for conversation and response storage.
    /// `None` means stateful features are disabled; all requests are proxied.
    pub db_url: Option<String>,
    pub model_aliases: HashMap<String, String>,
}

#[must_use]
pub fn normalize_base_url(url: &str) -> String {
    let mut s = url.trim_end_matches('/').to_owned();
    if s.ends_with("/v1") {
        s.truncate(s.len() - 3);
        s = s.trim_end_matches('/').to_owned();
    }
    s
}

#[must_use]
pub fn resolve_model_alias<S: BuildHasher>(model: &str, aliases: &HashMap<String, String, S>) -> String {
    aliases.get(model).cloned().unwrap_or_else(|| model.to_string())
}

/// Parse `alias=target` entries from CLI/env configuration.
///
/// # Errors
/// Returns an error when an entry is missing `=`, or either side is empty.
pub fn parse_model_aliases<I, S>(entries: I) -> Result<HashMap<String, String>, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut aliases = HashMap::new();
    for entry in entries {
        let entry = entry.as_ref().trim();
        if entry.is_empty() {
            continue;
        }
        let Some((alias, target)) = entry.split_once('=') else {
            return Err(format!("model alias '{entry}' must use alias=target"));
        };
        let alias = alias.trim();
        let target = target.trim();
        if alias.is_empty() || target.is_empty() {
            return Err(format!("model alias '{entry}' must have non-empty alias and target"));
        }
        aliases.insert(alias.to_string(), target.to_string());
    }
    Ok(aliases)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_trailing_v1() {
        assert_eq!(normalize_base_url("http://host:8000/v1"), "http://host:8000");
        assert_eq!(normalize_base_url("http://host:8000/v1/"), "http://host:8000");
    }

    #[test]
    fn no_v1_unchanged() {
        assert_eq!(normalize_base_url("http://host:8000"), "http://host:8000");
        assert_eq!(normalize_base_url("http://host:8000/"), "http://host:8000");
    }

    #[test]
    fn model_aliases_parse_and_resolve() {
        let aliases = parse_model_aliases(["codex-auto-review=Qwen/Qwen3"]).unwrap();
        assert_eq!(resolve_model_alias("codex-auto-review", &aliases), "Qwen/Qwen3");
        assert_eq!(resolve_model_alias("other", &aliases), "other");
    }

    #[test]
    fn model_aliases_reject_invalid_entries() {
        assert!(parse_model_aliases(["missing-separator"]).is_err());
        assert!(parse_model_aliases(["=target"]).is_err());
        assert!(parse_model_aliases(["alias="]).is_err());
    }
}
