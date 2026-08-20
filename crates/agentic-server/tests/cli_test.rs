use std::process::Command;

#[test]
fn missing_llm_api_base_error_mentions_config_environment_and_flag() {
    let home = tempfile::tempdir().expect("temporary Agentic API home");
    let output = Command::new(env!("CARGO_BIN_EXE_agentic-server"))
        .env("AGENTIC_API_HOME", home.path())
        .env_remove("LLM_API_BASE")
        .env_remove("OIDC_ISSUER")
        .env_remove("OIDC_AUDIENCE")
        .output()
        .expect("agentic-server must run");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("stderr must be UTF-8");
    assert!(
        stderr.contains("llm_api_base in config.toml, LLM_API_BASE, or --llm-api-base"),
        "unexpected error message: {stderr}"
    );
}

#[test]
fn help_does_not_expose_database_url_credentials() {
    let database_url = "postgresql://agentic:super-secret@db.example/agentic";
    let output = Command::new(env!("CARGO_BIN_EXE_agentic-server"))
        .env("DATABASE_URL", database_url)
        .arg("--help")
        .output()
        .expect("agentic-server help must run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout must be UTF-8");
    assert!(stdout.contains("DATABASE_URL"));
    assert!(!stdout.contains(database_url));
    assert!(!stdout.contains("super-secret"));
}
