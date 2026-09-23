//! The `tracing` subscriber stack: local log output plus, when traces are
//! exported, the `tracing-opentelemetry` bridge.
//!
//! The two layers are filtered independently: `RUST_LOG` only affects what
//! is printed locally, and the bridge only sees spans at `INFO` and above
//! whose target belongs to this repository's crates, so local log verbosity
//! never changes what is exported, dependency spans never leak into the
//! export, and sampling never changes what is logged.

use std::fmt;
use std::io;

use opentelemetry::trace::{SpanId, TraceContextExt, TraceId};
use opentelemetry_sdk::trace::SdkTracer;
use tracing::{Event, Level, Metadata, Subscriber};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::fmt::format::{Format, Full, Writer};
use tracing_subscriber::fmt::time::SystemTime;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields, MakeWriter};
use tracing_subscriber::layer::{Layer as _, SubscriberExt as _};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt as _;

use super::lifecycle::TelemetryError;

/// Local log filter used when `RUST_LOG` is unset.
///
/// The SDK and exporter report queue overflow and export failures through
/// `tracing` at `WARN` under their crate names, so those targets are included
/// by default: an operator whose collector is down must see it locally.
pub const DEFAULT_LOG_FILTER: &str =
    "agentic_server=info,agentic_core=info,opentelemetry_sdk=warn,opentelemetry-otlp=warn";

/// Crates whose spans may be exported. An allow-list rather than a deny-list:
/// dependencies such as `rmcp`, `hyper`, or the SDK itself declare their own
/// `INFO` spans, and none of those attributes have been reviewed for
/// cardinality or sensitivity. Matched against the leading `::` segment of a
/// span's target.
const EXPORTED_CRATES: &[&str] = &["agentic_server", "agentic_core", "agentic_praxis", "agentic_llm_d"];

/// Install the global subscriber.
///
/// Must be called at most once per process, before any request handling.
///
/// # Errors
///
/// Returns [`TelemetryError::Subscriber`] if a global subscriber is already
/// installed.
pub(crate) fn install(tracer: Option<SdkTracer>) -> Result<(), TelemetryError> {
    build_subscriber(tracer, log_filter_from_env(), io::stdout)
        .try_init()
        .map_err(TelemetryError::from)
}

/// `RUST_LOG`, falling back to [`DEFAULT_LOG_FILTER`].
fn log_filter_from_env() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_FILTER))
}

/// Assemble the layer stack without installing it.
///
/// Embedding applications and tests can scope the result with
/// [`tracing::subscriber::with_default`] or install it themselves; the
/// gateway binary installs it globally through [`crate::telemetry::init`].
pub fn build_subscriber<W>(tracer: Option<SdkTracer>, log_filter: EnvFilter, writer: W) -> impl Subscriber + Send + Sync
where
    W: for<'w> MakeWriter<'w> + Send + Sync + 'static,
{
    let fmt_layer = tracing_subscriber::fmt::layer()
        .event_format(CorrelatedFormat::new(tracer.is_some()))
        .with_writer(writer)
        .with_filter(log_filter);
    let otel_layer = tracer.map(|tracer| {
        tracing_opentelemetry::layer()
            .with_tracer(tracer)
            // Only declared span fields become attributes: no source location,
            // thread, `tracing` target/level, or busy/idle timings.
            .with_location(false)
            .with_threads(false)
            .with_target(false)
            .with_level(false)
            .with_tracked_inactivity(false)
            .with_filter(filter_fn(is_exported))
    });
    tracing_subscriber::registry().with(fmt_layer).with(otel_layer)
}

/// Bridge filter: spans only (events are reviewed and opted in individually
/// by later work), at `INFO` or above, from this repository's crates only.
fn is_exported(metadata: &Metadata<'_>) -> bool {
    metadata.is_span() && is_exported_level(*metadata.level()) && is_exported_target(metadata.target())
}

fn is_exported_level(level: Level) -> bool {
    level <= Level::INFO
}

fn is_exported_target(target: &str) -> bool {
    let crate_name = target.split("::").next().unwrap_or(target);
    EXPORTED_CRATES.contains(&crate_name)
}

/// Trace and span identifiers of the active OpenTelemetry span, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CorrelationIds {
    trace_id: TraceId,
    span_id: SpanId,
}

/// The bridge activates the OpenTelemetry context whenever a bridged span is
/// entered, so an event's correlation is simply the current context on the
/// emitting thread. Spans that are filtered away from the bridge inherit the
/// nearest exported ancestor, which is the span a reader can actually find.
fn current_correlation_ids() -> Option<CorrelationIds> {
    let context = opentelemetry::Context::current();
    let span = context.span();
    let span_context = span.span_context();
    span_context.is_valid().then(|| CorrelationIds {
        trace_id: span_context.trace_id(),
        span_id: span_context.span_id(),
    })
}

/// The default human-readable format with `trace_id`/`span_id` appended when
/// the event is emitted inside an exported span.
///
/// Without a bridge layer no span can carry an OpenTelemetry context, so the
/// formatter skips the context lookup entirely and delegates unchanged.
#[derive(Debug)]
struct CorrelatedFormat {
    inner: Format<Full, SystemTime>,
    correlate: bool,
}

impl CorrelatedFormat {
    fn new(correlate: bool) -> Self {
        Self {
            inner: Format::default(),
            correlate,
        }
    }
}

impl<S, N> FormatEvent<S, N> for CorrelatedFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(&self, ctx: &FmtContext<'_, S, N>, mut writer: Writer<'_>, event: &Event<'_>) -> fmt::Result {
        let Some(ids) = self.correlate.then(current_correlation_ids).flatten() else {
            return self.inner.format_event(ctx, writer, event);
        };
        // The inner format owns the line terminator, so render to a buffer
        // and splice the identifiers in before it. ANSI styling is not
        // propagated to the buffered writer; correlated lines are plain.
        let mut line = String::new();
        self.inner.format_event(ctx, Writer::new(&mut line), event)?;
        let line = line.trim_end_matches(['\n', '\r']);
        writeln!(writer, "{line} trace_id={} span_id={}", ids.trace_id, ids.span_id)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
    use tracing::{debug, info, info_span};

    use super::*;

    #[derive(Clone, Default)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    impl SharedBuffer {
        fn contents(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    impl io::Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'w> MakeWriter<'w> for SharedBuffer {
        type Writer = Self;

        fn make_writer(&'w self) -> Self::Writer {
            self.clone()
        }
    }

    /// The simple processor exports synchronously when a span closes, and the
    /// in-memory exporter clears itself on shutdown, so tests read spans
    /// before the provider is dropped.
    fn in_memory_tracer() -> (SdkTracerProvider, InMemorySpanExporter) {
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        (provider, exporter)
    }

    #[test]
    fn exported_levels_are_info_and_above() {
        assert!(is_exported_level(Level::ERROR));
        assert!(is_exported_level(Level::WARN));
        assert!(is_exported_level(Level::INFO));
        assert!(!is_exported_level(Level::DEBUG));
        assert!(!is_exported_level(Level::TRACE));
    }

    #[test]
    fn only_repository_crates_are_exported() {
        assert!(is_exported_target("agentic_core::executor::engine"));
        assert!(is_exported_target("agentic_server::telemetry::http"));
        assert!(is_exported_target("agentic_server"));
        for dependency in [
            "rmcp::service",
            "opentelemetry_sdk::trace",
            "opentelemetry-otlp",
            "hyper::proto::h1",
            "hyper_util::client::legacy",
            "reqwest",
            "tokio::task",
            "sqlx_core::pool",
            "agentic_server_evil::x",
        ] {
            assert!(!is_exported_target(dependency), "{dependency} must not be exported");
        }
    }

    #[test]
    fn dependency_spans_are_not_exported_even_at_info() {
        let (provider, exporter) = in_memory_tracer();
        let output = SharedBuffer::default();
        let subscriber = build_subscriber(Some(provider.tracer("test")), EnvFilter::new("info"), output.clone());
        tracing::subscriber::with_default(subscriber, || {
            let dependency = tracing::info_span!(target: "rmcp::service", "serve_inner");
            let _dependency = dependency.enter();
            let own = tracing::info_span!(target: "agentic_core::executor", "execute");
            let _own = own.enter();
        });
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "execute");
    }

    #[test]
    fn events_inside_exported_spans_carry_trace_and_span_ids() {
        let (provider, exporter) = in_memory_tracer();
        let output = SharedBuffer::default();
        let subscriber = build_subscriber(Some(provider.tracer("test")), EnvFilter::new("trace"), output.clone());

        tracing::subscriber::with_default(subscriber, || {
            info!("before any span");
            let span = info_span!("request");
            let _entered = span.enter();
            info!("inside span");
            debug!("debug inside span");
        });

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1, "exactly the INFO span is exported");
        let trace_id = spans[0].span_context.trace_id().to_string();
        let span_id = spans[0].span_context.span_id().to_string();
        assert_eq!(trace_id.len(), 32);
        assert_eq!(span_id.len(), 16);

        let logs = output.contents();
        let mut lines = logs.lines();
        let before = lines.next().unwrap();
        assert!(before.contains("before any span"), "{before}");
        assert!(!before.contains("trace_id="), "{before}");
        for expected in ["inside span", "debug inside span"] {
            let line = lines.next().unwrap();
            assert!(line.contains(expected), "{line}");
            assert!(
                line.ends_with(&format!("trace_id={trace_id} span_id={span_id}")),
                "{line}"
            );
        }
        assert_eq!(lines.next(), None);
    }

    #[test]
    fn debug_spans_are_not_exported_but_events_inherit_the_exported_parent() {
        let (provider, exporter) = in_memory_tracer();
        let output = SharedBuffer::default();
        let subscriber = build_subscriber(Some(provider.tracer("test")), EnvFilter::new("trace"), output.clone());

        tracing::subscriber::with_default(subscriber, || {
            let parent = info_span!("exported");
            let _parent = parent.enter();
            let child = tracing::debug_span!("local_only");
            let _child = child.enter();
            info!("nested event");
        });

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "exported");
        let span_id = spans[0].span_context.span_id().to_string();
        let logs = output.contents();
        assert!(logs.contains("nested event"), "{logs}");
        assert!(logs.trim_end().ends_with(&format!("span_id={span_id}")), "{logs}");
    }

    #[test]
    fn default_filter_surfaces_sdk_and_exporter_warnings() {
        let output = SharedBuffer::default();
        let subscriber = build_subscriber(None, EnvFilter::new(DEFAULT_LOG_FILTER), output.clone());
        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!(target: "opentelemetry_sdk", "sdk warn");
            tracing::warn!(target: "opentelemetry-otlp", "exporter warn");
            tracing::info!(target: "opentelemetry_sdk", "sdk info");
            tracing::info!(target: "opentelemetry-otlp", "exporter info");
            tracing::info!(target: "hyper_util::client", "transport info");
        });
        let logs = output.contents();
        assert!(logs.contains("sdk warn"), "{logs}");
        assert!(logs.contains("exporter warn"), "{logs}");
        assert!(!logs.contains("sdk info"), "{logs}");
        assert!(!logs.contains("exporter info"), "{logs}");
        assert!(!logs.contains("transport info"), "{logs}");
    }

    #[test]
    fn without_a_tracer_logs_are_plain() {
        let output = SharedBuffer::default();
        let subscriber = build_subscriber(None, EnvFilter::new("info"), output.clone());
        tracing::subscriber::with_default(subscriber, || {
            let span = info_span!("request");
            let _entered = span.enter();
            info!("no bridge installed");
        });
        let logs = output.contents();
        assert!(logs.contains("no bridge installed"), "{logs}");
        assert!(!logs.contains("trace_id="), "{logs}");
    }

    #[test]
    fn local_filter_does_not_affect_export() {
        let (provider, exporter) = in_memory_tracer();
        let output = SharedBuffer::default();
        let subscriber = build_subscriber(Some(provider.tracer("test")), EnvFilter::new("error"), output.clone());
        tracing::subscriber::with_default(subscriber, || {
            let span = info_span!("still_exported");
            let _entered = span.enter();
            info!("suppressed locally");
        });

        assert_eq!(exporter.get_finished_spans().unwrap().len(), 1);
        assert!(output.contents().is_empty());
    }
}
