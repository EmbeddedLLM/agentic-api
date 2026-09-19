//! Root HTTP server span and HTTP semantic-convention metrics.
//!
//! One span per request, `http.server.request`, parented on the caller's W3C
//! `traceparent` when one is present, opened before routing hands the
//! request to a handler and closed only when the response body has been
//! fully sent or dropped, so streaming responses are covered end to end. A
//! guard attached to the body records `http.server.request.duration` and
//! keeps `http.server.active_requests` balanced exactly once per request,
//! whichever way the body ends — including handler panics and failed
//! bodies, which also mark the span as errored, and requests cancelled before
//! a response exists (a timeout or runtime shutdown dropping the handler),
//! which are labelled but not treated as errors. Only bounded attributes are
//! recorded: no query strings, headers, client addresses, or original method
//! strings.

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::{MatchedPath, Request, State};
use axum::middleware::Next;
use axum::response::Response;
use http::{HeaderMap, Method, Version};
use http_body::{Body as HttpBody, Frame, SizeHint};
use opentelemetry::metrics::{Histogram, Meter, UpDownCounter};
use opentelemetry::propagation::Extractor;
use opentelemetry::{KeyValue, global};
use tracing::{Instrument, Span, field, info_span};
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

const INSTRUMENTATION_SCOPE: &str = "agentic_server";

const ATTR_METHOD: &str = "http.request.method";
const ATTR_ROUTE: &str = "http.route";
const ATTR_STATUS: &str = "http.response.status_code";
const ATTR_SCHEME: &str = "url.scheme";
const ATTR_ERROR_TYPE: &str = "error.type";
const ATTR_OTEL_STATUS: &str = "otel.status_code";

/// The gateway listens on plain HTTP; TLS, if any, terminates in front of it.
/// The scheme is therefore a property of the listener, never copied from the
/// request target (absolute-form targets can carry any client-chosen scheme).
const LISTENER_SCHEME: &str = "http";

/// Bounded `error.type` values for outcomes the middleware itself observes.
const ERROR_TYPE_PANIC: &str = "handler_panic";
const ERROR_TYPE_BODY: &str = "response_body";
const ERROR_TYPE_CANCELLED: &str = "cancelled";

/// Semantic-convention boundaries for `http.server.request.duration`,
/// extended past 10 s because streamed inference responses routinely run for
/// minutes.
const DURATION_BOUNDARIES: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0, 120.0, 300.0,
];

/// Instruments shared by every request; cheap to clone.
#[derive(Debug, Clone)]
pub struct HttpMetrics {
    duration: Histogram<f64>,
    active: UpDownCounter<i64>,
}

impl HttpMetrics {
    #[must_use]
    pub fn new(meter: &Meter) -> Self {
        Self {
            duration: meter
                .f64_histogram("http.server.request.duration")
                .with_unit("s")
                .with_description("Duration of HTTP server requests, from receipt to the end of the response body.")
                .with_boundaries(DURATION_BOUNDARIES.to_vec())
                .build(),
            active: meter
                .i64_up_down_counter("http.server.active_requests")
                .with_unit("{request}")
                .with_description("Number of HTTP requests currently being served.")
                .build(),
        }
    }

    /// Instruments bound to the globally registered meter provider.
    ///
    /// Must be called after the provider is installed (the gateway's
    /// `telemetry::init` runs before the router is built); with no provider
    /// installed the instruments are no-ops.
    #[must_use]
    pub fn from_global() -> Self {
        Self::new(&global::meter(INSTRUMENTATION_SCOPE))
    }
}

/// Bounded `http.request.method` label per the HTTP semantic conventions:
/// well-known methods verbatim, anything else collapsed to `_OTHER`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KnownMethod {
    Connect,
    Delete,
    Get,
    Head,
    Options,
    Patch,
    Post,
    Put,
    Trace,
    Other,
}

impl KnownMethod {
    fn as_str(self) -> &'static str {
        match self {
            Self::Connect => "CONNECT",
            Self::Delete => "DELETE",
            Self::Get => "GET",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
            Self::Patch => "PATCH",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Trace => "TRACE",
            Self::Other => "_OTHER",
        }
    }
}

impl From<&Method> for KnownMethod {
    fn from(method: &Method) -> Self {
        match *method {
            Method::CONNECT => Self::Connect,
            Method::DELETE => Self::Delete,
            Method::GET => Self::Get,
            Method::HEAD => Self::Head,
            Method::OPTIONS => Self::Options,
            Method::PATCH => Self::Patch,
            Method::POST => Self::Post,
            Method::PUT => Self::Put,
            Method::TRACE => Self::Trace,
            _ => Self::Other,
        }
    }
}

fn protocol_version(version: Version) -> &'static str {
    match version {
        Version::HTTP_09 => "0.9",
        Version::HTTP_10 => "1.0",
        Version::HTTP_11 => "1.1",
        Version::HTTP_2 => "2",
        Version::HTTP_3 => "3",
        _ => "unknown",
    }
}

/// Span name per the semantic conventions: `{method} {route}` when the
/// request matched a route template, otherwise the method alone.
fn span_name(method: KnownMethod, route: Option<&str>) -> String {
    match route {
        Some(route) => format!("{} {route}", method.as_str()),
        None => method.as_str().to_owned(),
    }
}

/// Axum middleware: wrap the whole request, including the response body, in
/// the root server span and the request metrics.
///
/// Install with `axum::middleware::from_fn_with_state(HttpMetrics, track_request)`
/// as the outermost `Router::layer` so that routing has already resolved the
/// matched path and rejected requests are still observed.
pub async fn track_request(State(metrics): State<HttpMetrics>, request: Request, next: Next) -> Response {
    let method = KnownMethod::from(request.method());
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|path| path.as_str().to_owned());

    let span = info_span!(
        "http.server.request",
        otel.name = %span_name(method, route.as_deref()),
        otel.kind = "server",
        otel.status_code = field::Empty,
        error.r#type = field::Empty,
        http.request.method = method.as_str(),
        http.route = field::Empty,
        http.response.status_code = field::Empty,
        url.scheme = LISTENER_SCHEME,
        network.protocol.version = protocol_version(request.version()),
    );
    if let Some(route) = &route {
        span.record(ATTR_ROUTE, route.as_str());
    }
    // A valid `traceparent` makes this span a child of the caller's and lets
    // the parent-based sampler honour the caller's decision; absent or invalid
    // headers leave it a root. `set_parent` only fails when no bridge layer is
    // installed, in which case there is nothing to attach to.
    let parent = global::get_text_map_propagator(|propagator| propagator.extract(&HeaderCarrier(request.headers())));
    let _ = span.set_parent(parent);

    let mut guard = RequestGuard::start(metrics, method, route);
    let mut outcome = SpanOutcome::pending(span.clone());
    let response = next.run(request).instrument(span.clone()).await;
    outcome.completed();

    let status = response.status();
    span.record(ATTR_STATUS, i64::from(status.as_u16()));
    if status.is_server_error() {
        span.record(ATTR_OTEL_STATUS, "ERROR");
    }
    guard.status = Some(status.as_u16());

    response.map(|body| {
        Body::new(TrackedBody {
            inner: body,
            span,
            _guard: guard,
        })
    })
}

/// Balances the active-request counter and records the duration exactly
/// once, on drop, so every way a body can end — completion, error, or the
/// client going away — is accounted for.
#[derive(Debug)]
struct RequestGuard {
    metrics: HttpMetrics,
    started: Instant,
    method: KnownMethod,
    route: Option<String>,
    status: Option<u16>,
}

impl RequestGuard {
    fn start(metrics: HttpMetrics, method: KnownMethod, route: Option<String>) -> Self {
        let guard = Self {
            metrics,
            started: Instant::now(),
            method,
            route,
            status: None,
        };
        guard.metrics.active.add(1, &guard.active_attributes());
        guard
    }

    fn active_attributes(&self) -> [KeyValue; 2] {
        [
            KeyValue::new(ATTR_METHOD, self.method.as_str()),
            KeyValue::new(ATTR_SCHEME, LISTENER_SCHEME),
        ]
    }

    fn duration_attributes(&self) -> Vec<KeyValue> {
        let mut attributes = self.active_attributes().to_vec();
        if let Some(route) = &self.route {
            attributes.push(KeyValue::new(ATTR_ROUTE, route.clone()));
        }
        if let Some(status) = self.status {
            attributes.push(KeyValue::new(ATTR_STATUS, i64::from(status)));
        }
        attributes
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.metrics.active.add(-1, &self.active_attributes());
        self.metrics
            .duration
            .record(self.started.elapsed().as_secs_f64(), &self.duration_attributes());
    }
}

/// Read-only view of the request headers for W3C context extraction.
struct HeaderCarrier<'a>(&'a HeaderMap);

impl Extractor for HeaderCarrier<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(http::HeaderName::as_str).collect()
    }
}

/// Records how a request ended when the handler never returned a response.
///
/// Two things drop this before [`SpanOutcome::completed`] is called: a panic
/// unwinding through the middleware, and the request future being dropped
/// while the handler is still pending — a timeout wrapping the handler, the
/// client going away before headers, or the runtime shutting down. Only the
/// former is a failure; the latter is labelled `cancelled` and leaves the
/// span status unset.
struct SpanOutcome {
    span: Span,
    completed: bool,
}

impl SpanOutcome {
    fn pending(span: Span) -> Self {
        Self { span, completed: false }
    }

    fn completed(&mut self) {
        self.completed = true;
    }
}

impl Drop for SpanOutcome {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        // A panicking handler drops the request future — and this with it —
        // while the panic is still unwinding (tokio drops a task's future
        // inside its unwind guard), so `panicking()` tells the two apart.
        if std::thread::panicking() {
            self.span.record(ATTR_OTEL_STATUS, "ERROR");
            self.span.record(ATTR_ERROR_TYPE, ERROR_TYPE_PANIC);
        } else {
            self.span.record(ATTR_ERROR_TYPE, ERROR_TYPE_CANCELLED);
        }
    }
}

/// Response body that keeps the request span alive and enters it on every
/// poll, so work done while streaming stays attributed to the request. A
/// body error marks the span; the guard is held only for its `Drop`, which
/// fires when this body does.
struct TrackedBody {
    inner: Body,
    span: Span,
    _guard: RequestGuard,
}

impl HttpBody for TrackedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // `Body`, `Span`, and the guard are all `Unpin`, so no projection is needed.
        let this = self.get_mut();
        let _entered = this.span.enter();
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        if let Poll::Ready(Some(Err(_))) = &polled {
            this.span.record(ATTR_OTEL_STATUS, "ERROR");
            this.span.record(ATTR_ERROR_TYPE, ERROR_TYPE_BODY);
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn methods_are_bounded_labels() {
        assert_eq!(KnownMethod::from(&Method::GET).as_str(), "GET");
        assert_eq!(KnownMethod::from(&Method::POST).as_str(), "POST");
        let custom = Method::from_bytes(b"PURGE").unwrap();
        assert_eq!(KnownMethod::from(&custom).as_str(), "_OTHER");
    }

    #[test]
    fn span_name_follows_semantic_conventions() {
        assert_eq!(
            span_name(KnownMethod::Post, Some("/v1/responses")),
            "POST /v1/responses"
        );
        assert_eq!(span_name(KnownMethod::Get, None), "GET");
    }

    #[test]
    fn protocol_versions_are_short_strings() {
        assert_eq!(protocol_version(Version::HTTP_11), "1.1");
        assert_eq!(protocol_version(Version::HTTP_2), "2");
    }
}
