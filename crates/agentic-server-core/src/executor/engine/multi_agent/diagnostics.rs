//! Metadata-only lifecycle diagnostics for cassette recording and operations.
use std::{future::Future, time::Instant};

use tracing::{info, warn};

use super::super::agent_turn::{RoundDecision, RoundResult};
use super::MultiAgentRun;
use crate::executor::error::ExecutorResult;
use crate::types::agent::{AgentIdentity, AgentMailContent};

pub(super) async fn observe_round(
    work: impl Future<Output = ExecutorResult<RoundResult>>,
) -> ExecutorResult<RoundResult> {
    let started = Instant::now();
    info!("agent round started");
    let result = work.await;
    let elapsed_ms = started.elapsed().as_millis();
    match &result {
        Ok(round) => {
            let decision = match &round.decision {
                RoundDecision::Continue => "continue",
                RoundDecision::Done => "done",
                RoundDecision::RequiresClientAction => "requires_client_action",
                RoundDecision::Incomplete(_) => "incomplete",
                RoundDecision::UpstreamTerminal => "upstream_terminal",
            };
            info!(elapsed_ms, decision, status = %round.payload.status,
                output_items = round.payload.output.len(),
                input_tokens = round.payload.usage.as_ref().map(|usage| usage.input_tokens),
                output_tokens = round.payload.usage.as_ref().map(|usage| usage.output_tokens),
                "agent round finished");
        }
        Err(error) => warn!(elapsed_ms, error_code = error.error_code(), "agent round failed"),
    }
    result
}

impl MultiAgentRun {
    pub(super) fn log_tree(&self, reason: &'static str) {
        for agent in self.registry.agents() {
            let context = &self.contexts[agent.identity];
            info!(response_id = %self.payload.id, agent = %agent.identity,
                turn = ?agent.turn, parent = agent.parent.map(AgentIdentity::as_str),
                state = ?agent.state, queued_messages = agent.queued_messages,
                rounds = context.stored.rounds, compacting = context.compacting,
                wait_deadline_ms = context.stored.wait.as_ref().map(|wait| wait.deadline_ms),
                reason, "multi-agent state");
        }
    }
}

pub(super) fn mail_kind(content: &AgentMailContent) -> &'static str {
    match content {
        AgentMailContent::Task(_) => "task",
        AgentMailContent::Message(_) => "message",
        AgentMailContent::TurnFinished(_) => "turn_finished",
    }
}
