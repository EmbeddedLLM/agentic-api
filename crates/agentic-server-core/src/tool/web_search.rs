use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::handler::{GatewayExecutor, GatewayToolEventPlan, ToolError, ToolHandler, ToolOutput};
use super::ownership::GatewayBinding;
use super::registry::{ToolEntry, ToolType};
use crate::types::io::output::{FunctionToolCall, WebSearchCall, WebSearchCallStatus, WebSearchSource};
use crate::types::io::{FunctionTool, OutputItem};
use crate::types::tools::{WebSearchContextSize, WebSearchToolParam};

const YOU_API_KEY: &str = "YOU_API_KEY";
const YOU_API_BASE_URL: &str = "YOU_API_BASE_URL";

pub(crate) type WebSearchExecutor =
    dyn GatewayExecutor<ToolParams = WebSearchToolParam, ExecutionParams = WebSearchToolParam>;

pub(crate) fn insert_web_search_entry(
    entries: &mut HashMap<String, ToolEntry>,
    params: &WebSearchToolParam,
    executor: Arc<WebSearchExecutor>,
) {
    entries.insert(
        "web_search".to_owned(),
        ToolEntry::gateway(
            ToolType::WebSearch,
            None,
            Some(GatewayBinding::new(executor, params.clone())),
        ),
    );
}

#[must_use]
pub(crate) fn web_search_function_tool() -> FunctionTool {
    FunctionTool {
        type_: "function".to_owned(),
        name: "web_search".to_owned(),
        description: Some(
            "Search the public web for current information and return structured web and news results.".to_owned(),
        ),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language web search query."
                },
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple independent search queries to run in parallel, instead of a single query."
                },
                "count": {
                    "type": "integer",
                    "description": "Maximum results per section, from 1 to 100."
                },
                "freshness": {
                    "type": "string",
                    "description": "Optional recency filter: day, week, month, year, or YYYY-MM-DDtoYYYY-MM-DD."
                },
                "country": {
                    "type": "string",
                    "description": "Optional ISO 3166-1 alpha-2 country code."
                },
                "language": {
                    "type": "string",
                    "description": "Optional BCP 47 language code."
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional strict allowlist of domains."
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional domain blocklist."
                }
            },
            "anyOf": [
                {"required": ["query"]},
                {"required": ["queries"]}
            ]
        })),
        strict: Some(false),
    }
}

#[must_use]
pub(crate) fn output_item(call: &FunctionToolCall, output: &ToolOutput, status: WebSearchCallStatus) -> OutputItem {
    let parsed_output = serde_json::from_str::<Value>(&output.output).ok();
    let queries = parsed_output
        .as_ref()
        .and_then(queries_from_value)
        .or_else(|| queries_from_arguments(&call.arguments))
        .unwrap_or_else(|| vec![String::new()]);
    let sources = parsed_output.as_ref().map(sources_from_output).unwrap_or_default();
    OutputItem::WebSearchCall(WebSearchCall::new(call_output_id(call), status, queries, sources))
}

#[must_use]
pub(crate) fn started_output_item(call: &FunctionToolCall) -> OutputItem {
    OutputItem::WebSearchCall(WebSearchCall::new(
        call_output_id(call),
        WebSearchCallStatus::InProgress,
        queries_from_arguments(&call.arguments).unwrap_or_else(|| vec![String::new()]),
        Vec::new(),
    ))
}

#[derive(Debug, Clone)]
pub struct WebSearchHandler {
    provider: Option<Arc<dyn WebSearchProvider>>,
}

impl WebSearchHandler {
    #[must_use]
    pub fn from_env(client: Arc<reqwest::Client>) -> Self {
        Self::from_values(
            client,
            std::env::var(YOU_API_KEY).ok(),
            std::env::var(YOU_API_BASE_URL).ok(),
        )
    }

    #[must_use]
    pub fn from_values(client: Arc<reqwest::Client>, api_key: Option<String>, base_url: Option<String>) -> Self {
        Self {
            provider: Some(Arc::new(YouSearchProvider::from_values(client, api_key, base_url))),
        }
    }

    #[must_use]
    pub fn with_api_key(client: Arc<reqwest::Client>, api_key: String, base_url: &str) -> Self {
        Self {
            provider: Some(Arc::new(YouSearchProvider::with_api_key(client, api_key, base_url))),
        }
    }

    /// Builds a handler usable only for shaping placeholder/error output
    /// (`ToolHandler::normalize`, `GatewayExecutor::started_output`/`public_output`)
    /// when no real provider is configured — `execute()` always fails.
    #[must_use]
    pub const fn spec_only() -> Self {
        Self { provider: None }
    }

    #[cfg(test)]
    fn with_provider(provider: Arc<dyn WebSearchProvider>) -> Self {
        Self {
            provider: Some(provider),
        }
    }

    async fn execute_search(
        &self,
        call_id: &str,
        arguments: &str,
        params: &WebSearchToolParam,
    ) -> Result<ToolOutput, ToolError> {
        let provider = self
            .provider
            .as_ref()
            .ok_or_else(|| ToolError::Config("web_search spec-only handler cannot execute tools".to_owned()))?;
        let args = WebSearchArguments::from_json(arguments)?;
        let queries = args.all_queries();
        let responses =
            futures::future::try_join_all(queries.iter().map(|query| provider.search(query, &args, params))).await?;

        let mut web = Vec::new();
        let mut news = Vec::new();
        let mut metadata = Vec::new();
        for response in responses {
            if let Some(results) = response.results.get("web").and_then(Value::as_array) {
                web.extend(results.iter().cloned());
            }
            if let Some(results) = response.results.get("news").and_then(Value::as_array) {
                news.extend(results.iter().cloned());
            }
            metadata.push(response.metadata);
        }
        let output = serde_json::to_string(&serde_json::json!({
            "query": queries[0],
            "queries": queries,
            "results": {"web": web, "news": news},
            "metadata": metadata
        }))
        .map_err(|e| ToolError::Execution(format!("failed to serialize web_search output: {e}")))?;

        Ok(ToolOutput {
            call_id: call_id.to_owned(),
            output,
        })
    }
}

trait WebSearchProvider: std::fmt::Debug + Send + Sync {
    fn search<'a>(
        &'a self,
        query: &'a str,
        args: &'a WebSearchArguments,
        config: &'a WebSearchToolParam,
    ) -> Pin<Box<dyn Future<Output = Result<WebSearchProviderResponse, ToolError>> + Send + 'a>>;
}

struct WebSearchProviderResponse {
    results: Value,
    metadata: Value,
}

#[derive(Debug, Clone)]
struct YouSearchProvider {
    client: Arc<reqwest::Client>,
    api_key: Option<String>,
    base_url: Option<String>,
}

impl YouSearchProvider {
    fn from_values(client: Arc<reqwest::Client>, api_key: Option<String>, base_url: Option<String>) -> Self {
        let api_key = api_key
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        let base_url = base_url.and_then(|value| clean_base_url(&value));
        Self {
            client,
            api_key,
            base_url,
        }
    }

    fn with_api_key(client: Arc<reqwest::Client>, api_key: String, base_url: &str) -> Self {
        Self {
            client,
            api_key: Some(api_key),
            base_url: clean_base_url(base_url),
        }
    }
}

impl WebSearchProvider for YouSearchProvider {
    fn search<'a>(
        &'a self,
        query: &'a str,
        args: &'a WebSearchArguments,
        config: &'a WebSearchToolParam,
    ) -> Pin<Box<dyn Future<Output = Result<WebSearchProviderResponse, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let api_key = self
                .api_key
                .as_deref()
                .ok_or_else(|| ToolError::Config(format!("{YOU_API_KEY} must be set to use the web_search tool")))?;
            let base_url = self.base_url.as_deref().ok_or_else(|| {
                ToolError::Config(format!("{YOU_API_BASE_URL} must be set to use the web_search tool"))
            })?;
            let request = YouSearchRequest::from_args_and_config(query, args, config)?;
            let resp = self
                .client
                .get(format!("{base_url}/v1/search"))
                .query(&request.query_params())
                .header("X-API-Key", api_key)
                .send()
                .await
                .map_err(|e| ToolError::Execution(format!("You.com search request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(ToolError::Execution(format!(
                    "You.com search returned {status}: {body}"
                )));
            }

            let response_text = resp
                .text()
                .await
                .map_err(|e| ToolError::Execution(format!("failed to read You.com search response: {e}")))?;
            let response: Value = serde_json::from_str(&response_text)
                .map_err(|e| ToolError::Execution(format!("You.com search returned invalid JSON: {e}")))?;
            Ok(WebSearchProviderResponse {
                results: response
                    .get("results")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"web": [], "news": []})),
                metadata: response.get("metadata").cloned().unwrap_or(Value::Null),
            })
        })
    }
}

impl ToolHandler for WebSearchHandler {
    type ToolParams = WebSearchToolParam;

    fn tool_type(&self) -> ToolType {
        ToolType::WebSearch
    }

    fn validate(&self, _params: &WebSearchToolParam) -> Result<(), ToolError> {
        Ok(())
    }

    fn normalize(&self, _params: &WebSearchToolParam) -> Vec<FunctionTool> {
        vec![web_search_function_tool()]
    }
}

impl GatewayExecutor for WebSearchHandler {
    type ExecutionParams = WebSearchToolParam;

    fn execute(
        &self,
        call_id: &str,
        tool_name: &str,
        arguments: &str,
        params: &WebSearchToolParam,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let call_id = call_id.to_owned();
        let tool_name = tool_name.to_owned();
        let arguments = arguments.to_owned();
        let params = params.clone();
        Box::pin(async move {
            if tool_name != "web_search" {
                return Err(ToolError::Config(format!(
                    "web_search handler cannot execute tool '{tool_name}'"
                )));
            }
            self.execute_search(&call_id, &arguments, &params).await
        })
    }

    fn supports_parallel_execution(&self) -> bool {
        true
    }

    fn plan_gateway_events(&self, call: &FunctionToolCall, _params: &WebSearchToolParam) -> GatewayToolEventPlan {
        GatewayToolEventPlan::new(Some(started_output_item(call)))
    }

    fn public_output(
        &self,
        call: &FunctionToolCall,
        output: &ToolOutput,
        status: WebSearchCallStatus,
        _params: &WebSearchToolParam,
    ) -> Option<OutputItem> {
        Some(output_item(call, output, status))
    }
}

#[derive(Debug, Deserialize)]
struct WebSearchArguments {
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    queries: Option<Vec<String>>,
    count: Option<u16>,
    freshness: Option<String>,
    country: Option<String>,
    language: Option<String>,
    safesearch: Option<String>,
    livecrawl: Option<String>,
    livecrawl_formats: Option<Vec<String>>,
    crawl_timeout: Option<u16>,
    include_domains: Option<Vec<String>>,
    exclude_domains: Option<Vec<String>>,
    boost_domains: Option<Vec<String>>,
}

impl WebSearchArguments {
    fn from_json(arguments: &str) -> Result<Self, ToolError> {
        let args = serde_json::from_str::<Self>(arguments)
            .map_err(|e| ToolError::Config(format!("web_search arguments must be valid JSON: {e}")))?;
        if args.all_queries().is_empty() {
            return Err(ToolError::Config(
                "web_search requires a non-empty query or queries".to_owned(),
            ));
        }
        Ok(args)
    }

    fn all_queries(&self) -> Vec<String> {
        let queries = clean_vec(self.queries.as_deref()).unwrap_or_default();
        if !queries.is_empty() {
            return queries;
        }
        clean_string(self.query.as_deref()).into_iter().collect()
    }
}

#[derive(Debug, Serialize)]
struct YouSearchRequest {
    query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    count: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    freshness: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safesearch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    livecrawl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    livecrawl_formats: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    crawl_timeout: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exclude_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    boost_domains: Option<Vec<String>>,
}

impl YouSearchRequest {
    fn query_params(&self) -> Vec<(String, String)> {
        let mut params = vec![("query".to_owned(), self.query.clone())];
        if let Some(count) = self.count {
            params.push(("count".to_owned(), count.to_string()));
        }
        if let Some(freshness) = &self.freshness {
            params.push(("freshness".to_owned(), freshness.clone()));
        }
        if let Some(country) = &self.country {
            params.push(("country".to_owned(), country.clone()));
        }
        if let Some(language) = &self.language {
            params.push(("language".to_owned(), language.clone()));
        }
        if let Some(safesearch) = &self.safesearch {
            params.push(("safesearch".to_owned(), safesearch.clone()));
        }
        if let Some(livecrawl) = &self.livecrawl {
            params.push(("livecrawl".to_owned(), livecrawl.clone()));
        }
        for format in self.livecrawl_formats.iter().flatten() {
            params.push(("livecrawl_formats".to_owned(), format.clone()));
        }
        if let Some(crawl_timeout) = self.crawl_timeout {
            params.push(("crawl_timeout".to_owned(), crawl_timeout.to_string()));
        }
        for domain in self.include_domains.iter().flatten() {
            params.push(("include_domains".to_owned(), domain.clone()));
        }
        for domain in self.exclude_domains.iter().flatten() {
            params.push(("exclude_domains".to_owned(), domain.clone()));
        }
        for domain in self.boost_domains.iter().flatten() {
            params.push(("boost_domains".to_owned(), domain.clone()));
        }
        params
    }

    fn from_args_and_config(
        query: &str,
        args: &WebSearchArguments,
        config: &WebSearchToolParam,
    ) -> Result<Self, ToolError> {
        let count = args
            .count
            .or_else(|| {
                config
                    .search_context_size
                    .map(WebSearchContextSize::default_count)
                    .map(u16::from)
            })
            .map(validate_count)
            .transpose()?;
        let crawl_timeout = args.crawl_timeout.map(validate_crawl_timeout).transpose()?;
        let config_domains = config
            .filters
            .as_ref()
            .and_then(|filters| clean_vec(filters.allowed_domains.as_deref()));
        let config_blocked_domains = config
            .filters
            .as_ref()
            .and_then(|filters| clean_vec(filters.blocked_domains.as_deref()));
        let include_domains = config_domains.or_else(|| clean_vec(args.include_domains.as_deref()));
        let exclude_domains = config_blocked_domains.or_else(|| clean_vec(args.exclude_domains.as_deref()));
        let boost_domains = clean_vec(args.boost_domains.as_deref());
        if include_domains.is_some() && (exclude_domains.is_some() || boost_domains.is_some()) {
            return Err(ToolError::Config(
                "include_domains cannot be combined with exclude_domains or boost_domains".to_owned(),
            ));
        }
        let country = config
            .user_location
            .as_ref()
            .and_then(|location| clean_string(location.country.as_deref()))
            .or_else(|| clean_string(args.country.as_deref()))
            .map(|value| value.to_ascii_uppercase());

        Ok(Self {
            query: query.trim().to_owned(),
            count,
            freshness: clean_string(args.freshness.as_deref()),
            country,
            language: clean_string(args.language.as_deref()),
            safesearch: clean_string(args.safesearch.as_deref()),
            livecrawl: clean_string(args.livecrawl.as_deref()),
            livecrawl_formats: clean_vec(args.livecrawl_formats.as_deref()),
            crawl_timeout,
            include_domains,
            exclude_domains,
            boost_domains,
        })
    }
}

fn validate_count(count: u16) -> Result<u8, ToolError> {
    if (1..=100).contains(&count) {
        Ok(u8::try_from(count).expect("validated web_search count must fit in u8"))
    } else {
        Err(ToolError::Config(
            "web_search count must be between 1 and 100".to_owned(),
        ))
    }
}

fn validate_crawl_timeout(timeout: u16) -> Result<u8, ToolError> {
    if (1..=60).contains(&timeout) {
        u8::try_from(timeout).map_err(|e| ToolError::Config(format!("invalid crawl_timeout: {e}")))
    } else {
        Err(ToolError::Config(
            "web_search crawl_timeout must be between 1 and 60".to_owned(),
        ))
    }
}

fn clean_string(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn clean_json_str(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn call_output_id(call: &FunctionToolCall) -> String {
    if let Some(suffix) = call.id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("ws_{suffix}");
    }
    if let Some(suffix) = call.call_id.strip_prefix("call_").filter(|suffix| !suffix.is_empty()) {
        return format!("ws_{suffix}");
    }
    crate::utils::uuid7_str("ws_")
}

fn queries_from_value(value: &Value) -> Option<Vec<String>> {
    let queries: Vec<String> = value
        .get("queries")?
        .as_array()?
        .iter()
        .filter_map(|item| clean_json_str(Some(item)))
        .collect();
    (!queries.is_empty()).then_some(queries)
}

fn queries_from_arguments(arguments: &str) -> Option<Vec<String>> {
    let args = serde_json::from_str::<Value>(arguments).ok()?;
    queries_from_value(&args).or_else(|| clean_json_str(args.get("query")).map(|query| vec![query]))
}

fn sources_from_output(output: &Value) -> Vec<WebSearchSource> {
    ["web", "news"]
        .into_iter()
        .filter_map(|section| output.get("results")?.get(section)?.as_array())
        .flat_map(|results| results.iter())
        .filter_map(source_from_result)
        .collect()
}

fn source_from_result(result: &Value) -> Option<WebSearchSource> {
    let url = clean_json_str(result.get("url"))?;
    Some(WebSearchSource {
        url,
        title: clean_json_str(result.get("title")),
    })
}

fn clean_base_url(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_end_matches('/');
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn clean_vec(values: Option<&[String]>) -> Option<Vec<String>> {
    let cleaned: Vec<String> = values
        .unwrap_or_default()
        .iter()
        .filter_map(|value| clean_string(Some(value.as_str())))
        .collect();
    (!cleaned.is_empty()).then_some(cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MockSearchProvider;

    impl WebSearchProvider for MockSearchProvider {
        fn search<'a>(
            &'a self,
            _query: &'a str,
            _args: &'a WebSearchArguments,
            _config: &'a WebSearchToolParam,
        ) -> Pin<Box<dyn Future<Output = Result<WebSearchProviderResponse, ToolError>> + Send + 'a>> {
            Box::pin(async move {
                Ok(WebSearchProviderResponse {
                    results: serde_json::json!({
                        "web": [
                            {
                                "url": "https://example.com/potato",
                                "title": "Potato"
                            }
                        ],
                        "news": []
                    }),
                    metadata: serde_json::json!({"provider": "mock"}),
                })
            })
        }
    }

    #[tokio::test]
    async fn web_search_handler_delegates_to_provider() {
        let handler = WebSearchHandler::with_provider(Arc::new(MockSearchProvider));
        let output = handler
            .execute(
                "call_search",
                "web_search",
                r#"{"query":" potato "}"#,
                &WebSearchToolParam::default(),
            )
            .await
            .unwrap();
        let body: Value = serde_json::from_str(&output.output).unwrap();
        assert_eq!(output.call_id, "call_search");
        assert_eq!(body["query"], "potato");
        assert_eq!(body["queries"], serde_json::json!(["potato"]));
        assert_eq!(body["metadata"][0]["provider"], "mock");
        assert_eq!(body["results"]["web"][0]["url"], "https://example.com/potato");
    }

    #[tokio::test]
    async fn web_search_handler_fans_out_multiple_queries() {
        let handler = WebSearchHandler::with_provider(Arc::new(MockSearchProvider));
        let output = handler
            .execute(
                "call_search",
                "web_search",
                r#"{"queries":["potato","tomato"]}"#,
                &WebSearchToolParam::default(),
            )
            .await
            .unwrap();
        let body: Value = serde_json::from_str(&output.output).unwrap();
        assert_eq!(body["query"], "potato");
        assert_eq!(body["queries"], serde_json::json!(["potato", "tomato"]));
        assert_eq!(body["results"]["web"].as_array().unwrap().len(), 2);
        assert_eq!(body["metadata"].as_array().unwrap().len(), 2);
    }
}
