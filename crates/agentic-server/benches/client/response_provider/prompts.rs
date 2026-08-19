use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::types::{ExpectedToolCall, PromptSpec, SessionSpec, ToolDefinition, TurnExpectation, Workload};

const TRANSPORT_TOOL_NAME: &str = "benchmark_step";

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub workload: Workload,
    pub seed: u64,
    pub sessions: usize,
    pub turns: usize,
    pub transport_rounds: usize,
    pub dataset_questions: Option<PathBuf>,
    pub dataset_answers: Option<PathBuf>,
    pub dataset_offset: usize,
}

#[derive(Debug, Error)]
pub enum PromptError {
    #[error("the tool-call workload requires --dataset-questions and --dataset-answers")]
    MissingDataset,
    #[error("failed to read benchmark dataset {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON on line {line} of {path}: {source}")]
    Json {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("BFCL answer is missing for case {0}")]
    MissingAnswer(String),
    #[error("BFCL case {0} has no user prompt")]
    MissingQuestion(String),
    #[error("requested {requested} BFCL cases at offset {offset}, but the dataset contains only {available}")]
    DatasetTooSmall {
        requested: usize,
        offset: usize,
        available: usize,
    },
}

#[derive(Debug, Deserialize)]
struct BfclQuestion {
    id: String,
    question: Vec<Vec<BfclMessage>>,
    #[serde(rename = "function")]
    functions: Vec<BfclFunction>,
}

#[derive(Debug, Deserialize)]
struct BfclMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct BfclFunction {
    name: String,
    #[serde(default)]
    description: String,
    parameters: Value,
}

#[derive(Debug, Deserialize)]
struct BfclAnswer {
    id: String,
    ground_truth: Vec<BTreeMap<String, BTreeMap<String, Vec<Value>>>>,
}

pub async fn generate(config: &GenerationConfig) -> Result<Vec<SessionSpec>, PromptError> {
    match config.workload {
        Workload::Transport => Ok(generate_transport(config)),
        Workload::ToolCall => generate_tool_calls(config).await,
        Workload::HistoryRehydration => Ok(generate_history(config)),
    }
}

fn generate_transport(config: &GenerationConfig) -> Vec<SessionSpec> {
    let tool = transport_tool();
    (0..config.sessions)
        .map(|session_index| {
            let prompts = (0..config.turns)
                .map(|turn_index| {
                    let run_id = format!("{:016X}", mixed_marker(config.seed, session_index, turn_index));
                    let calls = (1..=config.transport_rounds)
                        .map(|step| ExpectedToolCall {
                            name: TRANSPORT_TOOL_NAME.to_owned(),
                            arguments: BTreeMap::from([
                                ("run_id".to_owned(), vec![json!(run_id)]),
                                ("step".to_owned(), vec![json!(step)]),
                                ("total_steps".to_owned(), vec![json!(config.transport_rounds)]),
                            ]),
                        })
                        .collect();
                    let final_marker = transport_marker(&run_id, config.transport_rounds, config.transport_rounds);
                    let prompt = format!(
                        "This is a transport benchmark. Use only the `{TRANSPORT_TOOL_NAME}` function tool. Call it exactly \
                         {rounds} times, one call at a time and in order. For call N, pass run_id `{run_id}`, step N, \
                         and total_steps {rounds}. Wait for each tool call output before making the next call. After the \
                         last call, reply with exactly the marker returned by that last call and no other text.",
                        rounds = config.transport_rounds,
                    );
                    PromptSpec {
                        workload: Workload::Transport,
                        session_index,
                        turn_index,
                        prompt_id: format!("transport-s{session_index:03}-t{turn_index:03}"),
                        source_id: None,
                        prompt,
                        expectation: TurnExpectation::Transport { calls, final_marker },
                        tools: vec![tool.clone()],
                    }
                })
                .collect();
            SessionSpec {
                session_index,
                prompts,
            }
        })
        .collect()
}

async fn generate_tool_calls(config: &GenerationConfig) -> Result<Vec<SessionSpec>, PromptError> {
    let (Some(question_path), Some(answer_path)) = (&config.dataset_questions, &config.dataset_answers) else {
        return Err(PromptError::MissingDataset);
    };
    let questions: Vec<BfclQuestion> = read_jsonl(question_path).await?;
    let answers: Vec<BfclAnswer> = read_jsonl(answer_path).await?;
    let answers: HashMap<_, _> = answers
        .into_iter()
        .map(|answer| (answer.id, answer.ground_truth))
        .collect();
    let requested = config.sessions.saturating_mul(config.turns);
    let end = config.dataset_offset.saturating_add(requested);
    if end > questions.len() {
        return Err(PromptError::DatasetTooSmall {
            requested,
            offset: config.dataset_offset,
            available: questions.len(),
        });
    }

    let selected = &questions[config.dataset_offset..end];
    let mut sessions = Vec::with_capacity(config.sessions);
    for session_index in 0..config.sessions {
        let mut prompts = Vec::with_capacity(config.turns);
        for turn_index in 0..config.turns {
            let case = &selected[session_index * config.turns + turn_index];
            let ground_truth = answers
                .get(&case.id)
                .ok_or_else(|| PromptError::MissingAnswer(case.id.clone()))?;
            let prompt = bfcl_prompt(case)?;
            let tools = case.functions.iter().map(bfcl_tool).collect::<Vec<_>>();
            let calls = ground_truth
                .iter()
                .flat_map(|call_group| call_group.iter())
                .map(|(name, arguments)| ExpectedToolCall {
                    name: name.clone(),
                    arguments: arguments.clone(),
                })
                .collect();
            prompts.push(PromptSpec {
                workload: Workload::ToolCall,
                session_index,
                turn_index,
                prompt_id: format!("bfcl-{}", case.id),
                source_id: Some(case.id.clone()),
                prompt,
                expectation: TurnExpectation::ToolCalls { calls },
                tools,
            });
        }
        sessions.push(SessionSpec { session_index, prompts });
    }
    Ok(sessions)
}

fn generate_history(config: &GenerationConfig) -> Vec<SessionSpec> {
    (0..config.sessions)
        .map(|session_index| {
            let markers: Vec<String> = (0..=config.turns)
                .map(|turn_index| marker(config.seed, session_index, turn_index))
                .collect();
            let prompts = (0..config.turns)
                .map(|turn_index| {
                    let expected_marker = markers[turn_index].clone();
                    let next_marker = markers[turn_index + 1].clone();
                    let prompt = if turn_index == 0 {
                        initial_history_prompt(&expected_marker, &next_marker)
                    } else {
                        continuation_history_prompt(&next_marker)
                    };
                    PromptSpec {
                        workload: Workload::HistoryRehydration,
                        session_index,
                        turn_index,
                        prompt_id: format!("history-s{session_index:03}-t{turn_index:03}"),
                        source_id: None,
                        prompt,
                        expectation: TurnExpectation::Marker {
                            marker: expected_marker,
                        },
                        tools: Vec::new(),
                    }
                })
                .collect();
            SessionSpec { session_index, prompts }
        })
        .collect()
}

fn initial_history_prompt(expected_marker: &str, next_marker: &str) -> String {
    format!(
        "This is a state-continuation benchmark. Reply with exactly `{expected_marker}` and no other text. Privately \
         remember this next-turn secret: `{next_marker}`; do not print it yet."
    )
}

fn continuation_history_prompt(next_marker: &str) -> String {
    format!(
        "Recall the next-turn secret from my immediately preceding request; I am intentionally not restating its \
         value. Reply with exactly the recalled value and no other text. \
         Privately remember this new next-turn secret: `{next_marker}`; do not print it yet."
    )
}

fn bfcl_prompt(case: &BfclQuestion) -> Result<String, PromptError> {
    let user_text = case
        .question
        .iter()
        .flatten()
        .filter(|message| message.role == "user")
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    if user_text.is_empty() {
        return Err(PromptError::MissingQuestion(case.id.clone()));
    }
    Ok(format!(
        "This is a BFCL tool-calling evaluation. Call only the provided function tool or tools needed to satisfy the \
         request. Do not invent arguments. Return the required function call or calls.\n\nUser request: {user_text}"
    ))
}

fn bfcl_tool(function: &BfclFunction) -> ToolDefinition {
    let mut parameters = function.parameters.clone();
    normalize_schema(&mut parameters);
    let parameters = parameters.as_object().cloned().unwrap_or_else(|| {
        Map::from_iter([
            ("type".to_owned(), Value::String("object".to_owned())),
            ("properties".to_owned(), Value::Object(Map::new())),
        ])
    });
    ToolDefinition {
        name: function.name.clone(),
        description: function.description.clone(),
        parameters,
    }
}

fn normalize_schema(value: &mut Value) {
    match value {
        Value::Object(object) => {
            if let Some(Value::String(kind)) = object.get_mut("type") {
                *kind = match kind.as_str() {
                    "dict" => "object",
                    "float" => "number",
                    "list" | "tuple" => "array",
                    other => other,
                }
                .to_owned();
            }
            for child in object.values_mut() {
                normalize_schema(child);
            }
        }
        Value::Array(values) => {
            for child in values {
                normalize_schema(child);
            }
        }
        _ => {}
    }
}

fn transport_tool() -> ToolDefinition {
    ToolDefinition {
        name: TRANSPORT_TOOL_NAME.to_owned(),
        description: "Advance exactly one round of a deterministic transport benchmark.".to_owned(),
        parameters: Map::from_iter([
            ("type".to_owned(), json!("object")),
            (
                "properties".to_owned(),
                json!({
                    "run_id": {"type": "string"},
                    "step": {"type": "integer", "minimum": 1},
                    "total_steps": {"type": "integer", "minimum": 1}
                }),
            ),
            ("required".to_owned(), json!(["run_id", "step", "total_steps"])),
            ("additionalProperties".to_owned(), json!(false)),
        ]),
    }
}

pub fn transport_marker(run_id: &str, step: usize, total_steps: usize) -> String {
    format!("TRANSPORT_MARKER_{run_id}_{step}_OF_{total_steps}")
}

async fn read_jsonl<T>(path: &Path) -> Result<Vec<T>, PromptError>
where
    T: for<'de> Deserialize<'de>,
{
    let text = tokio::fs::read_to_string(path)
        .await
        .map_err(|source| PromptError::Read {
            path: path.to_owned(),
            source,
        })?;
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|source| PromptError::Json {
                path: path.to_owned(),
                line: index + 1,
                source,
            })
        })
        .collect()
}

fn marker(seed: u64, session_index: usize, turn_index: usize) -> String {
    format!("HISTORY_MARKER_{:016X}", mixed_marker(seed, session_index, turn_index))
}

fn mixed_marker(seed: u64, session_index: usize, turn_index: usize) -> u64 {
    let session = u64::try_from(session_index).unwrap_or(u64::MAX);
    let turn = u64::try_from(turn_index).unwrap_or(u64::MAX);
    splitmix64(seed ^ session.rotate_left(17) ^ turn.rotate_left(39))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
