#[path = "support/multi_agent_contract.rs"]
mod multi_agent_contract;
mod support;

use multi_agent_contract::{ComparisonPolicy, RecordedSession, assert_multi_agent_contract};

#[test]
fn reference_cassettes_have_consistent_multi_agent_contract() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cassettes/multi_agent");
    for suffix in [
        "review-gpt-5.6-sol-nonstreaming",
        "review-gpt-5.6-sol-streaming",
        "proposals-gpt-5.6-sol-nonstreaming",
        "proposals-gpt-5.6-sol-streaming",
        "mixed-tools-gpt-5.6-sol-streaming",
    ] {
        let path = directory.join(format!("multi-agent-openai-reference-{suffix}.yaml"));
        RecordedSession::load(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    }
}
use std::path::Path;

#[test]
#[ignore = "requires independently recorded OpenAI and gateway cassettes"]
fn compare_recorded_multi_agent_contract() {
    let reference = std::env::var("MULTI_AGENT_REFERENCE_CASSETTE").expect("set reference cassette path");
    let gateway = std::env::var("MULTI_AGENT_GATEWAY_CASSETTE").expect("set gateway cassette path");
    let reference = RecordedSession::load(Path::new(&reference)).unwrap();
    let gateway = RecordedSession::load(Path::new(&gateway)).unwrap();
    assert_multi_agent_contract(
        &reference,
        &gateway,
        &ComparisonPolicy {
            require_reference_tool_kinds: true,
        },
    )
    .unwrap();
}
