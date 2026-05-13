use agentic_api::config::{DbDialect, RuntimeConfig};

pub fn test_config(upstream_url: &str) -> RuntimeConfig {
    RuntimeConfig {
        llm_api_base: upstream_url.to_owned(),
        openai_api_key: Some("env-upstream-key".to_owned()),
        gateway_host: "127.0.0.1".to_owned(),
        gateway_port: 0,
        gateway_workers: 1,
        upstream_ready_timeout_s: 5.0,
        upstream_ready_interval_s: 0.1,
        db_url: "sqlite://:memory:?mode=memory".to_owned(),
        db_dialect: DbDialect::Sqlite,
        response_store_enabled: false,
        log_model_messages: false,
    }
}

pub fn test_config_no_key(upstream_url: &str) -> RuntimeConfig {
    RuntimeConfig {
        openai_api_key: None,
        ..test_config(upstream_url)
    }
}
