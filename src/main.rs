use clap::{Parser, Subcommand};

use agentic_api::config::{RuntimeConfig, normalize_base_url};
use agentic_api::entrypoints::serve as server;

#[derive(Parser)]
#[command(name = "agentic-api", about = "Stateful API gateway for vLLM Responses API")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(long)]
    llm_api_base: Option<String>,

    #[command(flatten)]
    config: RuntimeConfig,
}

#[derive(Subcommand)]
enum Commands {
    Serve {
        model: String,

        #[arg(long, default_value_t = 8000)]
        port: u16,

        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        vllm_args: Vec<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "agentic_api=info".parse().expect("valid filter")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        None => {
            let base = cli.llm_api_base.ok_or(
                "standalone mode requires --llm-api-base; use `agentic-api serve <model>` for integrated mode",
            )?;
            let config = cli.config.finalize(&base).expect("invalid configuration");
            server::run(config).await
        }
        Some(Commands::Serve { model, port, vllm_args }) => {
            let base = normalize_base_url(&format!("http://127.0.0.1:{port}"));
            let config = cli.config.finalize(&base).expect("invalid configuration");

            let mut args = vec!["--model".to_owned(), model];
            args.push("--port".to_owned());
            args.push(port.to_string());
            args.extend(vllm_args);

            server::run_with_vllm(config, args).await
        }
    }
}
