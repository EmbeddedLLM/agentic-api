use std::time::Duration;

use clap::{Args, Parser, Subcommand};

use agentic_core::DatabaseBackend;
use agentic_core::config::{
    Config, DEFAULT_POSTGRES_ACQUIRE_TIMEOUT_SECONDS, DEFAULT_POSTGRES_IDLE_TIMEOUT_SECONDS,
    DEFAULT_POSTGRES_LOCK_TIMEOUT_SECONDS, DEFAULT_POSTGRES_MAX_CONNECTIONS, DEFAULT_POSTGRES_MAX_LIFETIME_SECONDS,
    DEFAULT_POSTGRES_MIGRATION_TIMEOUT_SECONDS, DEFAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS,
    DEFAULT_SQLITE_JOURNAL_SIZE_LIMIT_BYTES, DEFAULT_SQLITE_MAX_CONNECTIONS, DEFAULT_SQLITE_MMAP_SIZE_BYTES,
    PostgresConfig, SqliteConfig, SqliteTempStore, normalize_base_url,
};
use agentic_core::error::Error;
use agentic_server::auth::OidcConfig;

mod server;

#[derive(Args, Clone)]
struct CommonArgs {
    #[arg(long, env = "OPENAI_API_KEY", hide_env_values = true, global = true)]
    openai_api_key: Option<String>,

    /// OIDC issuer for optional inbound bearer-token authentication.
    #[arg(long, env = "OIDC_ISSUER", global = true)]
    oidc_issuer: Option<String>,

    /// Required bearer-token audience when `OIDC_ISSUER` is configured.
    #[arg(long, env = "OIDC_AUDIENCE", global = true)]
    oidc_audience: Option<String>,

    #[arg(long, env = "GATEWAY_HOST", default_value = "0.0.0.0", global = true)]
    gateway_host: String,

    #[arg(long, env = "GATEWAY_PORT", default_value_t = 9000, global = true)]
    gateway_port: u16,

    #[arg(long, default_value_t = 600.0, global = true)]
    llm_ready_timeout_s: f64,

    #[arg(long, default_value_t = 2.0, global = true)]
    llm_ready_interval_s: f64,

    /// Skip the upstream /health readiness probe. Useful for hosted OpenAI-compatible providers.
    #[arg(long, env = "SKIP_LLM_READY_CHECK", default_value_t = false, global = true)]
    skip_llm_ready_check: bool,

    /// `SQLite` or `PostgreSQL` URL for conversation and response storage.
    /// Defaults to a local `SQLite` file.
    #[arg(
        long,
        env = "DATABASE_URL",
        hide_env_values = true,
        default_value = "sqlite://./agentic_api.db",
        global = true
    )]
    db_url: String,
}

fn oidc_config_from_values(
    issuer: Option<&str>,
    audience: Option<&str>,
) -> Result<Option<OidcConfig>, server::ServerError> {
    match (issuer, audience) {
        (None, None) => Ok(None),
        (Some(issuer), Some(audience)) => Ok(Some(OidcConfig::new(issuer, audience)?)),
        _ => Err(Error::Config("OIDC_ISSUER and OIDC_AUDIENCE must be configured together".to_owned()).into()),
    }
}

#[derive(Parser)]
#[command(name = "agentic-server", about = "Stateful API gateway for vLLM Responses API")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(long, env = "LLM_API_BASE")]
    llm_api_base: Option<String>,

    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Subcommand)]
enum Commands {
    /// Spawn vLLM and run the gateway in the foreground
    Serve {
        /// Model name or path
        model: String,

        /// vLLM server port
        #[arg(long, default_value_t = 8000)]
        port: u16,

        /// Additional arguments passed through to vLLM
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        llm_args: Vec<String>,
    },
}

fn parse_env_u64(name: &str, default: u64) -> Result<u64, Error> {
    parse_env_u64_value(name, std::env::var(name), default)
}

fn parse_env_u64_value(name: &str, value: Result<String, std::env::VarError>, default: u64) -> Result<u64, Error> {
    match value {
        Ok(value) => value
            .parse::<u64>()
            .map_err(|e| Error::Config(format!("{name} must be an unsigned integer: {e}"))),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(Error::Config(format!("failed to read {name}: {e}"))),
    }
}

fn parse_env_u32(name: &str, default: u32) -> Result<u32, Error> {
    parse_env_u32_value(name, std::env::var(name), default)
}

fn parse_env_u32_value(name: &str, value: Result<String, std::env::VarError>, default: u32) -> Result<u32, Error> {
    match value {
        Ok(value) => {
            let parsed = value
                .parse::<u32>()
                .map_err(|e| Error::Config(format!("{name} must be a positive integer: {e}")))?;
            if parsed == 0 {
                return Err(Error::Config(format!("{name} must be greater than 0")));
            }
            Ok(parsed)
        }
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(Error::Config(format!("failed to read {name}: {e}"))),
    }
}

fn parse_env_duration(name: &str, default_seconds: u64) -> Result<Duration, Error> {
    parse_env_duration_value(name, std::env::var(name), default_seconds)
}

fn parse_env_duration_value(
    name: &str,
    value: Result<String, std::env::VarError>,
    default_seconds: u64,
) -> Result<Duration, Error> {
    let seconds = parse_env_u64_value(name, value, default_seconds)?;
    if seconds == 0 {
        return Err(Error::Config(format!("{name} must be greater than 0")));
    }
    Ok(Duration::from_secs(seconds))
}

fn parse_env_optional_duration(name: &str, default_seconds: u64) -> Result<Option<Duration>, Error> {
    parse_env_optional_duration_value(name, std::env::var(name), default_seconds)
}

fn parse_env_optional_duration_value(
    name: &str,
    value: Result<String, std::env::VarError>,
    default_seconds: u64,
) -> Result<Option<Duration>, Error> {
    let seconds = parse_env_u64_value(name, value, default_seconds)?;
    Ok((seconds > 0).then(|| Duration::from_secs(seconds)))
}

fn parse_env_temp_store() -> Result<SqliteTempStore, Error> {
    parse_env_temp_store_value(std::env::var("SQLITE_TEMP_STORE"))
}

fn parse_env_temp_store_value(value: Result<String, std::env::VarError>) -> Result<SqliteTempStore, Error> {
    match value {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "default" | "0" => Ok(SqliteTempStore::Default),
            "file" | "1" => Ok(SqliteTempStore::File),
            "memory" | "2" => Ok(SqliteTempStore::Memory),
            _ => Err(Error::Config(
                "SQLITE_TEMP_STORE must be one of default, file, memory, 0, 1, or 2".to_owned(),
            )),
        },
        Err(std::env::VarError::NotPresent) => Ok(SqliteTempStore::default()),
        Err(e) => Err(Error::Config(format!("failed to read SQLITE_TEMP_STORE: {e}"))),
    }
}

fn sqlite_config_from_env() -> Result<SqliteConfig, Error> {
    Ok(SqliteConfig {
        max_connections: parse_env_u32("SQLITE_MAX_CONNECTIONS", DEFAULT_SQLITE_MAX_CONNECTIONS)?,
        journal_size_limit_bytes: parse_env_u64(
            "SQLITE_JOURNAL_SIZE_LIMIT_BYTES",
            DEFAULT_SQLITE_JOURNAL_SIZE_LIMIT_BYTES,
        )?,
        temp_store: parse_env_temp_store()?,
        mmap_size_bytes: parse_env_u64("SQLITE_MMAP_SIZE_BYTES", DEFAULT_SQLITE_MMAP_SIZE_BYTES)?,
    })
}

fn postgres_config_from_env() -> Result<PostgresConfig, Error> {
    Ok(PostgresConfig {
        max_connections: parse_env_u32("POSTGRES_MAX_CONNECTIONS", DEFAULT_POSTGRES_MAX_CONNECTIONS)?,
        acquire_timeout: parse_env_duration(
            "POSTGRES_ACQUIRE_TIMEOUT_SECONDS",
            DEFAULT_POSTGRES_ACQUIRE_TIMEOUT_SECONDS,
        )?,
        lock_timeout: parse_env_duration("POSTGRES_LOCK_TIMEOUT_SECONDS", DEFAULT_POSTGRES_LOCK_TIMEOUT_SECONDS)?,
        migration_timeout: parse_env_duration(
            "POSTGRES_MIGRATION_TIMEOUT_SECONDS",
            DEFAULT_POSTGRES_MIGRATION_TIMEOUT_SECONDS,
        )?,
        statement_timeout: parse_env_duration(
            "POSTGRES_STATEMENT_TIMEOUT_SECONDS",
            DEFAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS,
        )?,
        idle_timeout: parse_env_optional_duration(
            "POSTGRES_IDLE_TIMEOUT_SECONDS",
            DEFAULT_POSTGRES_IDLE_TIMEOUT_SECONDS,
        )?,
        max_lifetime: parse_env_optional_duration(
            "POSTGRES_MAX_LIFETIME_SECONDS",
            DEFAULT_POSTGRES_MAX_LIFETIME_SECONDS,
        )?,
    })
}

fn database_configs_from_env(database_url: &str) -> Result<(PostgresConfig, SqliteConfig), Error> {
    let backend = DatabaseBackend::from_url(database_url)
        .map_err(|error| Error::Config(format!("invalid DATABASE_URL: {error}")))?;
    match backend {
        DatabaseBackend::Postgres => Ok((postgres_config_from_env()?, SqliteConfig::default())),
        DatabaseBackend::Sqlite => Ok((PostgresConfig::default(), sqlite_config_from_env()?)),
        DatabaseBackend::Other => Ok((PostgresConfig::default(), SqliteConfig::default())),
    }
}

fn build_config(llm_api_base: String, common: &CommonArgs) -> Result<Config, Error> {
    let (postgres, sqlite) = database_configs_from_env(&common.db_url)?;
    Ok(Config {
        llm_api_base,
        openai_api_key: common.openai_api_key.clone(),
        llm_ready_timeout_s: common.llm_ready_timeout_s,
        llm_ready_interval_s: common.llm_ready_interval_s,
        skip_llm_ready_check: common.skip_llm_ready_check,
        db_url: Some(common.db_url.clone()),
        postgres,
        sqlite,
    })
}

#[tokio::main]
async fn main() -> Result<(), server::ServerError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "agentic_server=info,agentic_core=info".parse().expect("valid filter")),
        )
        .init();

    let Cli {
        command,
        llm_api_base,
        common,
    } = Cli::parse();
    let oidc_config = oidc_config_from_values(common.oidc_issuer.as_deref(), common.oidc_audience.as_deref())?;

    match command {
        None => {
            let base = llm_api_base.ok_or_else(|| {
                Error::Config(
                    "standalone mode requires LLM_API_BASE (or --llm-api-base); use `agentic-server serve <model>` for integrated mode"
                        .to_owned(),
                )
            })?;
            let config = build_config(normalize_base_url(&base), &common)?;
            server::run(config, &common.gateway_host, common.gateway_port, oidc_config).await
        }
        Some(Commands::Serve { model, port, llm_args }) => {
            if llm_api_base.is_some() {
                return Err(Error::Config(
                    "--llm-api-base is only valid in standalone mode; remove it when using `serve`".to_owned(),
                )
                .into());
            }
            let config = build_config(normalize_base_url(&format!("http://127.0.0.1:{port}")), &common)?;
            let mut args = vec!["--model".to_owned(), model];
            args.push("--port".to_owned());
            args.push(port.to_string());
            args.extend(llm_args);
            server::run_with_llm(config, &common.gateway_host, common.gateway_port, args, oidc_config).await
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use clap::{CommandFactory, Parser};

    use super::{
        Cli, Commands, database_configs_from_env, oidc_config_from_values, parse_env_duration_value,
        parse_env_optional_duration_value, parse_env_temp_store_value, parse_env_u32_value, parse_env_u64_value,
    };
    use agentic_core::config::{
        DEFAULT_POSTGRES_ACQUIRE_TIMEOUT_SECONDS, DEFAULT_POSTGRES_IDLE_TIMEOUT_SECONDS,
        DEFAULT_POSTGRES_LOCK_TIMEOUT_SECONDS, DEFAULT_POSTGRES_MAX_CONNECTIONS,
        DEFAULT_POSTGRES_MIGRATION_TIMEOUT_SECONDS, DEFAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS,
        DEFAULT_SQLITE_MAX_CONNECTIONS, SqliteTempStore,
    };

    #[test]
    fn serve_uses_common_args_before_subcommand() {
        let cli = Cli::parse_from(["agentic-server", "--llm-ready-timeout-s", "0.1", "serve", "model-a"]);
        assert!((cli.common.llm_ready_timeout_s - 0.1).abs() < f64::EPSILON);
        assert!(matches!(cli.command, Some(Commands::Serve { .. })));
    }

    #[test]
    fn serve_uses_common_args_after_subcommand() {
        let cli = Cli::parse_from(["agentic-server", "serve", "--llm-ready-timeout-s", "0.1", "model-a"]);
        assert!((cli.common.llm_ready_timeout_s - 0.1).abs() < f64::EPSILON);
        assert!(matches!(cli.command, Some(Commands::Serve { .. })));
    }

    #[test]
    fn skip_llm_ready_check_can_be_set_from_cli() {
        let cli = Cli::parse_from([
            "agentic-server",
            "--llm-api-base",
            "http://localhost:8000",
            "--skip-llm-ready-check",
        ]);
        assert!(cli.common.skip_llm_ready_check);
    }

    #[test]
    fn oidc_configuration_requires_issuer_and_audience_together() {
        assert!(oidc_config_from_values(None, None).expect("disabled OIDC").is_none());
        assert!(oidc_config_from_values(Some("https://issuer.example"), None).is_err());
        assert!(oidc_config_from_values(None, Some("agentic-api")).is_err());
        assert!(
            oidc_config_from_values(Some("https://issuer.example"), Some("agentic-api"))
                .expect("complete OIDC configuration")
                .is_some()
        );
    }

    #[test]
    fn container_runtime_options_are_bound_to_environment_variables() {
        let command = Cli::command();

        for (argument, expected_env) in [
            ("llm_api_base", "LLM_API_BASE"),
            ("gateway_host", "GATEWAY_HOST"),
            ("gateway_port", "GATEWAY_PORT"),
            ("oidc_issuer", "OIDC_ISSUER"),
            ("oidc_audience", "OIDC_AUDIENCE"),
        ] {
            let env = command
                .get_arguments()
                .find(|arg| arg.get_id() == argument)
                .and_then(clap::Arg::get_env)
                .unwrap_or_else(|| panic!("{argument} must be configurable through {expected_env}"));

            assert_eq!(env, expected_env);
        }
    }

    #[test]
    fn sqlite_tuning_is_env_only_not_cli() {
        let mut help = Vec::new();
        Cli::command().write_long_help(&mut help).expect("render help");
        let help = String::from_utf8(help).expect("help is utf8");

        assert!(!help.contains("--sqlite-journal-size-limit-bytes"));
        assert!(!help.contains("--sqlite-max-connections"));
        assert!(!help.contains("--sqlite-temp-store"));
        assert!(!help.contains("--sqlite-mmap-size-bytes"));

        assert!(
            Cli::try_parse_from([
                "agentic-server",
                "--llm-api-base",
                "http://localhost:8000",
                "--sqlite-temp-store",
                "memory",
            ])
            .is_err()
        );
    }

    #[test]
    fn sqlite_tuning_env_parser_uses_defaults_and_rejects_invalid_values() {
        assert_eq!(
            parse_env_u32_value(
                "SQLITE_MAX_CONNECTIONS",
                Err(std::env::VarError::NotPresent),
                DEFAULT_SQLITE_MAX_CONNECTIONS
            )
            .expect("default value"),
            DEFAULT_SQLITE_MAX_CONNECTIONS
        );
        assert_eq!(
            parse_env_u32_value(
                "SQLITE_MAX_CONNECTIONS",
                Ok("6".to_owned()),
                DEFAULT_SQLITE_MAX_CONNECTIONS
            )
            .expect("parsed value"),
            6
        );
        assert!(
            parse_env_u32_value(
                "SQLITE_MAX_CONNECTIONS",
                Ok("0".to_owned()),
                DEFAULT_SQLITE_MAX_CONNECTIONS
            )
            .is_err()
        );
        assert!(
            parse_env_u32_value(
                "SQLITE_MAX_CONNECTIONS",
                Ok("not-a-number".to_owned()),
                DEFAULT_SQLITE_MAX_CONNECTIONS
            )
            .is_err()
        );

        assert_eq!(
            parse_env_u64_value("SQLITE_MMAP_SIZE_BYTES", Err(std::env::VarError::NotPresent), 1_024)
                .expect("default value"),
            1_024
        );
        assert_eq!(
            parse_env_u64_value("SQLITE_MMAP_SIZE_BYTES", Ok("4096".to_owned()), 1_024).expect("parsed value"),
            4_096
        );
        assert!(parse_env_u64_value("SQLITE_MMAP_SIZE_BYTES", Ok("not-a-number".to_owned()), 1_024).is_err());

        assert_eq!(
            parse_env_temp_store_value(Err(std::env::VarError::NotPresent)).expect("default temp store"),
            SqliteTempStore::Memory
        );
        assert_eq!(
            parse_env_temp_store_value(Ok("file".to_owned())).expect("file temp store"),
            SqliteTempStore::File
        );
        assert_eq!(
            parse_env_temp_store_value(Ok("2".to_owned())).expect("memory temp store"),
            SqliteTempStore::Memory
        );
        assert!(parse_env_temp_store_value(Ok("invalid".to_owned())).is_err());
    }

    #[test]
    fn postgres_timeout_parser_uses_defaults_and_allows_disabling_recycling() {
        assert_eq!(
            parse_env_duration_value(
                "POSTGRES_ACQUIRE_TIMEOUT_SECONDS",
                Err(std::env::VarError::NotPresent),
                DEFAULT_POSTGRES_ACQUIRE_TIMEOUT_SECONDS,
            )
            .expect("default acquire timeout"),
            Duration::from_secs(DEFAULT_POSTGRES_ACQUIRE_TIMEOUT_SECONDS)
        );
        assert_eq!(
            parse_env_duration_value(
                "POSTGRES_ACQUIRE_TIMEOUT_SECONDS",
                Ok("9".to_owned()),
                DEFAULT_POSTGRES_ACQUIRE_TIMEOUT_SECONDS,
            )
            .expect("explicit acquire timeout"),
            Duration::from_secs(9)
        );
        assert!(
            parse_env_duration_value(
                "POSTGRES_ACQUIRE_TIMEOUT_SECONDS",
                Ok("0".to_owned()),
                DEFAULT_POSTGRES_ACQUIRE_TIMEOUT_SECONDS,
            )
            .is_err()
        );
        assert_eq!(
            parse_env_optional_duration_value(
                "POSTGRES_IDLE_TIMEOUT_SECONDS",
                Ok("0".to_owned()),
                DEFAULT_POSTGRES_IDLE_TIMEOUT_SECONDS,
            )
            .expect("disabled idle timeout"),
            None
        );
        assert_eq!(
            parse_env_optional_duration_value(
                "POSTGRES_IDLE_TIMEOUT_SECONDS",
                Ok("45".to_owned()),
                DEFAULT_POSTGRES_IDLE_TIMEOUT_SECONDS,
            )
            .expect("explicit idle timeout"),
            Some(Duration::from_secs(45))
        );
        assert_eq!(
            parse_env_duration_value(
                "POSTGRES_LOCK_TIMEOUT_SECONDS",
                Err(std::env::VarError::NotPresent),
                DEFAULT_POSTGRES_LOCK_TIMEOUT_SECONDS,
            )
            .expect("default lock timeout"),
            Duration::from_secs(DEFAULT_POSTGRES_LOCK_TIMEOUT_SECONDS)
        );
        assert_eq!(
            parse_env_duration_value(
                "POSTGRES_MIGRATION_TIMEOUT_SECONDS",
                Err(std::env::VarError::NotPresent),
                DEFAULT_POSTGRES_MIGRATION_TIMEOUT_SECONDS,
            )
            .expect("default migration timeout"),
            Duration::from_secs(DEFAULT_POSTGRES_MIGRATION_TIMEOUT_SECONDS)
        );
        assert_eq!(
            parse_env_duration_value(
                "POSTGRES_STATEMENT_TIMEOUT_SECONDS",
                Err(std::env::VarError::NotPresent),
                DEFAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS,
            )
            .expect("default statement timeout"),
            Duration::from_secs(DEFAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS)
        );
        assert!(
            parse_env_u32_value(
                "POSTGRES_MAX_CONNECTIONS",
                Ok("0".to_owned()),
                DEFAULT_POSTGRES_MAX_CONNECTIONS,
            )
            .is_err()
        );
    }

    #[test]
    fn database_config_rejects_an_invalid_url() {
        let error = database_configs_from_env("not a database URL").expect_err("invalid URL must be rejected");
        assert!(error.to_string().contains("invalid DATABASE_URL"));
    }
}
