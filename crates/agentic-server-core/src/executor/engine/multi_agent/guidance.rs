//! Fresh, model-only ownership context; never part of the durable agent history.
use super::MultiAgentRun;
use crate::executor::multi_agent::collaboration::instructions;
use crate::types::agent::AgentTurnKey;
use crate::types::io::{InputMessage, InputMessageContent};

impl MultiAgentRun {
    pub(super) fn round_guidance(&self, turn: &AgentTurnKey) -> InputMessage {
        let context = &self.contexts[&turn.agent];
        let children = self
            .registry
            .agents()
            .filter(|agent| agent.parent == Some(&turn.agent))
            .map(|agent| format!("{}: {:?}", agent.identity, agent.state))
            .collect::<Vec<_>>();
        let ownership = if children.is_empty() && turn.agent.is_root() {
            "You have no direct children yet. Delegate independent tasks when useful; wait only after successful delegation."
                .to_owned()
        } else if children.is_empty() {
            "You have no direct children. No child result is outstanding. Complete your assigned work yourself."
                .to_owned()
        } else {
            format!(
                "Your direct children and their current states:\n{}",
                children.join("\n")
            )
        };
        let task = if turn.agent.is_root() {
            "Your assignment is the user's overall request."
        } else {
            &context.stored.last_task
        };
        InputMessage {
            role: "developer".into(),
            content: InputMessageContent::Text(format!(
                "{}\n\nCurrent agent ownership (gateway state):\n{}\n\nYour current assignment:\n{}",
                instructions(&turn.agent, self.limit),
                ownership,
                task
            )),
            ..Default::default()
        }
    }
}
