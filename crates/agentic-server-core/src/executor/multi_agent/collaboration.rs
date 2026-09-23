//! Model-facing collaboration declarations and opaque public projection.

use base64::Engine;
use ring::{
    aead,
    rand::{SecureRandom, SystemRandom},
};

use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::types::agent::AgentIdentity;
use crate::types::agent_commands::AgentCommand;
use crate::types::io::{AgentAttribution, FunctionTool};
use crate::types::request_response::UpstreamTool;
use crate::utils::common::serialize_to_string;

pub(in crate::executor) fn attribution(identity: &AgentIdentity) -> AgentAttribution {
    AgentAttribution {
        agent_name: identity.as_str().to_owned(),
    }
}

pub(in crate::executor) fn instructions(identity: &AgentIdentity, max_subagents: usize) -> String {
    format!(
        "You are `{identity}`, an agent in a team working on the user's task. \
         The root agent /root synthesizes the final answer. Each agent has its own context and the same tools. \
         Use spawn_agent for independent bounded tasks; send_message queues information without activating idle agents; \
         followup_task activates a non-root agent; wait_agent waits for mailbox updates; interrupt_agent interrupts \
         active work while retaining context; list_agents reports the tree. Targets can be child names or canonical paths. \
         There are {max_subagents} active subagent slots across all descendants, excluding /root. \
         Spawn returns an acknowledgement; the answer arrives later in a mailbox message. \
         Subagent final answers are delivered to their parent. Continue useful work while agents run. \
         Function calls and local shell calls are executed by the client; their outputs may arrive in a later response."
    )
}

pub(in crate::executor) fn tools() -> Vec<UpstreamTool> {
    use serde_json::json;
    [
        (
            "spawn_agent",
            "Create a subagent for an independent task. fork_turns is all, none, or a positive integer string.",
            json!({"task_name":{"type":"string"},"message":{"type":"string"},"fork_turns":{"type":"string"}}),
            vec!["task_name", "message"],
        ),
        (
            "send_message",
            "Queue a message without starting an idle agent.",
            json!({"target":{"type":"string"},"message":{"type":"string"}}),
            vec!["target", "message"],
        ),
        (
            "followup_task",
            "Assign work to an existing non-root agent, starting or resuming its turn.",
            json!({"target":{"type":"string"},"message":{"type":"string"}}),
            vec!["target", "message"],
        ),
        (
            "wait_agent",
            "Wait for a mailbox update. timeout_ms is between 10000 and 3600000, default 30000.",
            json!({"timeout_ms":{"type":"integer","minimum":10_000,"maximum":3_600_000}}),
            vec![],
        ),
        (
            "interrupt_agent",
            "Interrupt another agent's active turn, retaining its context.",
            json!({"target":{"type":"string"}}),
            vec!["target"],
        ),
        (
            "list_agents",
            "List agent paths, statuses and their most recent assigned task.",
            json!({}),
            vec![],
        ),
    ]
    .into_iter()
    .map(|(name, description, properties, required)| {
        UpstreamTool::Function(FunctionTool {
            type_: "function".into(),
            name: name.to_owned(),
            description: Some(description.to_owned()),
            parameters: Some(
                json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),
            ),
            strict: Some(false),
        })
    })
    .collect()
}

/// A response-local key protects public transcript content. Durable continuation
/// uses the private canonical tree, never decrypted client-supplied transcript.
pub(in crate::executor) struct TranscriptSealer(aead::LessSafeKey);

impl TranscriptSealer {
    pub(in crate::executor) fn new() -> ExecutorResult<Self> {
        let mut key = [0; 32];
        SystemRandom::new().fill(&mut key).map_err(|_| crypto_error())?;
        let key = aead::UnboundKey::new(&aead::AES_256_GCM, &key).map_err(|_| crypto_error())?;
        Ok(Self(aead::LessSafeKey::new(key)))
    }

    pub(in crate::executor) fn seal(&self, text: &str) -> ExecutorResult<String> {
        let mut nonce = [0; aead::NONCE_LEN];
        SystemRandom::new().fill(&mut nonce).map_err(|_| crypto_error())?;
        let mut ciphertext = text.as_bytes().to_vec();
        self.0
            .seal_in_place_append_tag(
                aead::Nonce::assume_unique_for_key(nonce),
                aead::Aad::from(b"agentic-api/multi-agent/v1"),
                &mut ciphertext,
            )
            .map_err(|_| crypto_error())?;
        let mut envelope = nonce.to_vec();
        envelope.append(&mut ciphertext);
        Ok(format!(
            "enc_{}",
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(envelope)
        ))
    }

    pub(in crate::executor) fn arguments(&self, command: &AgentCommand) -> ExecutorResult<String> {
        let mut public = command.clone();
        match &mut public {
            AgentCommand::Spawn(task) => task.message = self.seal(&task.message)?,
            AgentCommand::Send(task) | AgentCommand::Followup(task) => task.message = self.seal(&task.message)?,
            AgentCommand::Wait(_) | AgentCommand::Interrupt(_) | AgentCommand::List(_) => {}
        }
        serialize_to_string(&public).map_err(ExecutorError::JsonError)
    }
}

fn crypto_error() -> ExecutorError {
    ExecutorError::StreamError("could not seal the multi-agent transcript".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::agent_commands::SpawnAgent;
    #[test]
    fn public_arguments_hide_task_text_and_use_distinct_nonces() {
        let sealer = TranscriptSealer::new().unwrap();
        let command = AgentCommand::Spawn(SpawnAgent {
            task_name: "review".into(),
            message: "private task".into(),
            fork_turns: "all".into(),
        });
        let first = sealer.arguments(&command).unwrap();
        let second = sealer.arguments(&command).unwrap();
        assert!(!first.contains("private task"));
        assert!(first.contains("review"));
        assert_ne!(first, second);
    }
}
