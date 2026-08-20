use clap::{
    Args, Parser, Subcommand, ValueEnum,
    builder::{Styles, styling::AnsiColor},
};

const fn brand_styles() -> Styles {
    Styles::styled()
        .header(AnsiColor::BrightCyan.on_default().bold())
        .usage(AnsiColor::BrightBlue.on_default().bold())
        .literal(AnsiColor::BrightYellow.on_default().bold())
        .placeholder(AnsiColor::BrightMagenta.on_default())
        .valid(AnsiColor::BrightGreen.on_default())
}

pub const DEFAULT_DATABASE_URL: &str = "sqlite://./agentic_api.db";

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Harness {
    Codex,
    Claude,
}

#[derive(Debug, Parser)]
#[command(
    name = "agentic",
    about = "Agentic API — local agent gateway for Claude Code and Codex",
    version,
    styles = brand_styles(),
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Start Agentic API and launch a coding harness
    Run {
        #[command(subcommand)]
        harness: HarnessCommand,
    },
    /// Start Agentic API without launching a harness
    Serve(ServeOptions),
    /// Validate the local Agentic API session prerequisites
    Validate(ValidateOptions),
}

#[derive(Debug, Subcommand)]
pub enum HarnessCommand {
    /// Launch Codex with an isolated provider configuration
    Codex(HarnessOptions),
    /// Launch Claude Code with an isolated gateway environment
    Claude(HarnessOptions),
}

#[derive(Args, Clone, Debug)]
pub struct HarnessOptions {
    #[command(flatten)]
    pub source: SourceOptions,

    #[command(flatten)]
    pub common: CommonOptions,

    /// Arguments forwarded to the selected harness after `--`
    #[arg(last = true, allow_hyphen_values = true)]
    pub harness_args: Vec<String>,
}

#[derive(Args, Clone, Debug)]
pub struct ServeOptions {
    #[command(flatten)]
    pub source: SourceOptions,

    #[command(flatten)]
    pub common: CommonOptions,
}

#[derive(Args, Clone, Debug)]
pub struct ValidateOptions {
    #[command(flatten)]
    pub source: SourceOptions,

    #[command(flatten)]
    pub common: CommonOptions,

    /// Also verify a harness binary without launching it
    #[arg(long, value_enum)]
    pub harness: Option<Harness>,
}

#[derive(Args, Clone, Debug)]
pub struct SourceOptions {
    /// Connect to an already-running OpenAI-compatible upstream
    #[arg(long, required_unless_present = "model")]
    pub upstream: Option<String>,

    /// Start vLLM with this model before starting the gateway
    #[arg(long, required_unless_present = "upstream")]
    pub model: Option<String>,

    /// vLLM port when starting a model
    #[arg(long, default_value_t = 8000)]
    pub llm_port: u16,
}

#[derive(Args, Clone, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct CommonOptions {
    /// Gateway bind host
    #[arg(long, default_value = "127.0.0.1", env = "GATEWAY_HOST")]
    pub gateway_host: String,

    /// Gateway bind port
    #[arg(long, default_value_t = 3000, env = "GATEWAY_PORT")]
    pub gateway_port: u16,

    /// `SQLite` or `PostgreSQL` storage URL
    #[arg(long, default_value = DEFAULT_DATABASE_URL, env = "DATABASE_URL", hide_env_values = true)]
    pub database_url: String,

    /// API key forwarded to the gateway and harness when configured
    #[arg(long, env = "OPENAI_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// Skip the upstream readiness probe
    #[arg(long, default_value_t = false)]
    pub skip_llm_ready_check: bool,

    /// Upstream readiness timeout in seconds
    #[arg(long, default_value_t = 600.0, value_parser = parse_timeout_seconds)]
    pub llm_ready_timeout_s: f64,

    /// Upstream readiness poll interval in seconds
    #[arg(long, default_value_t = 2.0, value_parser = parse_interval_seconds)]
    pub llm_ready_interval_s: f64,

    /// Suppress lifecycle output
    #[arg(long)]
    pub quiet: bool,

    /// Skip harness permission prompts and sandbox restrictions
    #[arg(long)]
    pub yolo: bool,

    /// Disable ANSI color output
    #[arg(long)]
    pub no_color: bool,
}

fn parse_timeout_seconds(value: &str) -> Result<f64, String> {
    let value = value
        .parse::<f64>()
        .map_err(|error| format!("invalid timeout in seconds: {error}"))?;
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err("timeout must be a finite, non-negative number of seconds".to_owned())
    }
}

fn parse_interval_seconds(value: &str) -> Result<f64, String> {
    let value = value
        .parse::<f64>()
        .map_err(|error| format!("invalid interval in seconds: {error}"))?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err("interval must be a finite, positive number of seconds".to_owned())
    }
}

impl Default for CommonOptions {
    fn default() -> Self {
        Self {
            gateway_host: "127.0.0.1".to_owned(),
            gateway_port: 3000,
            database_url: DEFAULT_DATABASE_URL.to_owned(),
            api_key: None,
            skip_llm_ready_check: false,
            llm_ready_timeout_s: 600.0,
            llm_ready_interval_s: 2.0,
            quiet: false,
            yolo: false,
            no_color: false,
        }
    }
}

impl HarnessCommand {
    #[must_use]
    pub fn harness(&self) -> Harness {
        match self {
            Self::Codex(_) => Harness::Codex,
            Self::Claude(_) => Harness::Claude,
        }
    }

    #[must_use]
    pub fn options(&self) -> &HarnessOptions {
        match self {
            Self::Codex(options) | Self::Claude(options) => options,
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command, DEFAULT_DATABASE_URL, HarnessCommand};

    #[test]
    fn run_codex_uses_sqlite_by_default_and_preserves_arguments() {
        let cli = Cli::try_parse_from([
            "agentic",
            "run",
            "codex",
            "--model",
            "Qwen/test",
            "--",
            "exec",
            "inspect this repo",
        ])
        .expect("valid CLI");

        let Command::Run { harness } = cli.command else {
            panic!("expected run command");
        };
        assert!(matches!(harness, HarnessCommand::Codex(_)));
        let options = harness.options();
        assert_eq!(options.source.model.as_deref(), Some("Qwen/test"));
        assert_eq!(options.common.database_url, DEFAULT_DATABASE_URL);
        assert_eq!(options.harness_args, ["exec", "inspect this repo"]);
    }

    #[test]
    fn run_claude_accepts_an_explicit_postgres_database() {
        let cli = Cli::try_parse_from([
            "agentic",
            "run",
            "claude",
            "--upstream",
            "http://127.0.0.1:8000",
            "--database-url",
            "postgresql://user:secret@localhost/agentic",
        ])
        .expect("valid CLI");

        let Command::Run { harness } = cli.command else {
            panic!("expected run command");
        };
        assert!(matches!(harness, HarnessCommand::Claude(_)));
        let options = harness.options();
        assert_eq!(options.source.upstream.as_deref(), Some("http://127.0.0.1:8000"));
        assert_eq!(
            options.common.database_url,
            "postgresql://user:secret@localhost/agentic"
        );
    }

    #[test]
    fn run_accepts_upstream_with_an_explicit_model_name() {
        let result = Cli::try_parse_from([
            "agentic",
            "run",
            "codex",
            "--model",
            "Qwen/test",
            "--upstream",
            "http://127.0.0.1:8000",
        ]);

        let cli = result.expect("valid CLI");
        let Command::Run { harness } = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(harness.options().source.model.as_deref(), Some("Qwen/test"));
    }

    #[test]
    fn run_accepts_yolo_mode() {
        let cli =
            Cli::try_parse_from(["agentic", "run", "claude", "--model", "Qwen/test", "--yolo"]).expect("valid CLI");

        let Command::Run { harness } = cli.command else {
            panic!("expected run command");
        };
        assert!(harness.options().common.yolo);
    }
}
