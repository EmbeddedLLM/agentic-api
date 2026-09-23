//! Provider lifecycle behaviour against a local OTLP/HTTP stub: nothing is
//! contacted when telemetry is disabled, exported data carries the configured
//! service identity, and shutdown stays inside its deadline no matter how the
//! collector behaves.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use opentelemetry::trace::{Span as _, Tracer as _};

use agentic_server::telemetry::{TelemetryConfig, TelemetryError, init_providers};

// Only the OTLP stub is used from the shared helpers here.
#[allow(dead_code)]
mod common;
use common::otlp_stub::{OtlpStub, StubMode, service_name};

fn enabled_config(endpoint: &str) -> TelemetryConfig {
    TelemetryConfig::from_lookup(|name| match name {
        "OTEL_TRACES_EXPORTER" | "OTEL_METRICS_EXPORTER" => Some("otlp".to_owned()),
        "OTEL_SERVICE_NAME" => Some("agentic-api-test".to_owned()),
        _ => None,
    })
    .unwrap()
    .with_otlp_endpoint(endpoint)
}

#[tokio::test]
async fn disabled_configuration_builds_nothing_and_contacts_nothing() {
    let (endpoint, stub) = OtlpStub::spawn(StubMode::Accept).await;
    let config = TelemetryConfig::disabled().with_otlp_endpoint(&endpoint);

    let (guard, handles) = init_providers(&config).unwrap();
    assert!(!guard.is_enabled());
    assert!(handles.tracer.is_none());
    assert!(handles.meter.is_none());

    guard.shutdown(Duration::from_secs(1)).await.unwrap();
    assert_eq!(stub.connections.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exports_carry_the_configured_service_identity() {
    let (endpoint, stub) = OtlpStub::spawn(StubMode::Accept).await;
    let config = enabled_config(&endpoint).with_otlp_timeout(Duration::from_secs(2));

    let (guard, handles) = init_providers(&config).unwrap();
    assert!(guard.is_enabled());

    let tracer = handles.tracer.expect("traces enabled");
    let mut span = tracer.start("spike.span");
    span.end();
    let meter = handles.meter.expect("metrics enabled");
    meter.u64_counter("spike.counter").build().add(1, &[]);

    guard.shutdown(Duration::from_secs(5)).await.unwrap();

    let trace_exports = stub.trace_exports().await;
    let export = &trace_exports[0];
    assert_eq!(
        service_name(export.resource_spans[0].resource.as_ref()),
        Some("agentic-api-test")
    );
    let span_names: Vec<&str> = export.resource_spans[0]
        .scope_spans
        .iter()
        .flat_map(|scope| scope.spans.iter().map(|span| span.name.as_str()))
        .collect();
    assert_eq!(span_names, ["spike.span"]);

    let metric_exports = stub.metric_exports().await;
    let export = &metric_exports[0];
    assert_eq!(
        service_name(export.resource_metrics[0].resource.as_ref()),
        Some("agentic-api-test")
    );
    let metric_names: Vec<&str> = export.resource_metrics[0]
        .scope_metrics
        .iter()
        .flat_map(|scope| scope.metrics.iter().map(|metric| metric.name.as_str()))
        .collect();
    assert_eq!(metric_names, ["spike.counter"]);
}

/// `OTEL_EXPORTER_OTLP_COMPRESSION=gzip` must build (the exporter needs its
/// `gzip-http` feature for that) and the collector must receive bodies it
/// can inflate back into the same protobuf payloads.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gzip_compressed_exports_reach_the_collector() {
    let (endpoint, stub) = OtlpStub::spawn(StubMode::Accept).await;
    let config = TelemetryConfig::from_lookup(|name| match name {
        "OTEL_TRACES_EXPORTER" | "OTEL_METRICS_EXPORTER" => Some("otlp".to_owned()),
        "OTEL_EXPORTER_OTLP_COMPRESSION" => Some("gzip".to_owned()),
        _ => None,
    })
    .unwrap()
    .with_otlp_endpoint(&endpoint)
    .with_otlp_timeout(Duration::from_secs(2));

    let (guard, handles) = init_providers(&config).expect("gzip is compiled into the exporter");
    let tracer = handles.tracer.expect("traces enabled");
    let mut span = tracer.start("compressed.span");
    span.end();
    let meter = handles.meter.expect("metrics enabled");
    meter.u64_counter("compressed.counter").build().add(1, &[]);

    guard.shutdown(Duration::from_secs(5)).await.unwrap();

    let requests = stub.connections.load(Ordering::SeqCst);
    assert!(requests >= 2, "one export per signal, got {requests}");
    assert_eq!(
        stub.gzip_requests.load(Ordering::SeqCst),
        requests,
        "every export body is gzip-encoded"
    );
    let span_names: Vec<String> = stub
        .trace_exports()
        .await
        .iter()
        .flat_map(|export| export.resource_spans.iter())
        .flat_map(|resource| resource.scope_spans.iter())
        .flat_map(|scope| scope.spans.iter().map(|span| span.name.clone()))
        .collect();
    assert_eq!(span_names, ["compressed.span"]);
    let metric_names: Vec<String> = stub
        .metric_exports()
        .await
        .iter()
        .flat_map(|export| export.resource_metrics.iter())
        .flat_map(|resource| resource.scope_metrics.iter())
        .flat_map(|scope| scope.metrics.iter().map(|metric| metric.name.clone()))
        .collect();
    assert_eq!(metric_names, ["compressed.counter"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_completes_when_the_collector_times_out() {
    let (endpoint, _stub) = OtlpStub::spawn(StubMode::Hang).await;
    let config = enabled_config(&endpoint).with_otlp_timeout(Duration::from_millis(200));
    let (guard, handles) = init_providers(&config).unwrap();
    let mut span = handles.tracer.unwrap().start("spike.span");
    span.end();
    handles.meter.unwrap().u64_counter("spike.counter").build().add(1, &[]);

    let started = Instant::now();
    let result = guard.shutdown(Duration::from_secs(3)).await;
    let elapsed = started.elapsed();

    assert!(elapsed < Duration::from_secs(3), "shutdown took {elapsed:?}");
    match result {
        Ok(()) | Err(TelemetryError::ProviderShutdown { .. }) => {}
        Err(other) => panic!("unexpected shutdown outcome: {other}"),
    }
}

/// An unresponsive collector combined with a long exporter timeout: the
/// guard must give up at its own deadline rather than wait for the exporter.
///
/// Uses a dedicated runtime so the still-running blocking shutdown does not
/// hold the test open; `main` stops its runtime the same way.
#[test]
fn shutdown_deadline_is_honoured_with_an_unresponsive_collector() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let deadline = Duration::from_millis(500);

    let (result, elapsed) = runtime.block_on(async {
        let (endpoint, _stub) = OtlpStub::spawn(StubMode::Hang).await;
        let config = enabled_config(&endpoint).with_otlp_timeout(Duration::from_secs(60));
        let (guard, handles) = init_providers(&config).unwrap();
        let mut span = handles.tracer.unwrap().start("spike.span");
        span.end();
        // A pending metric makes the final export block on the collector, so
        // the blocking shutdown provably outlives the guard's deadline.
        handles.meter.unwrap().u64_counter("spike.counter").build().add(1, &[]);

        let started = Instant::now();
        let result = guard.shutdown(deadline).await;
        (result, started.elapsed())
    });

    assert!(
        matches!(result, Err(TelemetryError::ShutdownTimeout { deadline: reported }) if reported == deadline),
        "expected a deadline error, got {result:?}"
    );
    assert!(elapsed >= deadline, "returned before the deadline: {elapsed:?}");
    assert!(elapsed < deadline * 3, "overshot the deadline: {elapsed:?}");

    runtime.shutdown_background();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_an_enabled_guard_inside_the_runtime_does_not_panic() {
    let (endpoint, _stub) = OtlpStub::spawn(StubMode::Accept).await;
    let config = enabled_config(&endpoint).with_otlp_timeout(Duration::from_millis(200));
    let (guard, handles) = init_providers(&config).unwrap();
    let mut span = handles.tracer.unwrap().start("spike.span");
    span.end();
    drop(handles.meter);
    drop(guard);
}

/// The path `main` uses after its runtime has stopped: no Tokio context, a
/// plain thread does the shutdown, and the deadline is still enforced.
#[test]
fn blocking_shutdown_honours_its_deadline_outside_a_runtime() {
    let stub_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let (endpoint, _stub) = stub_runtime.block_on(OtlpStub::spawn(StubMode::Hang));
    let config = enabled_config(&endpoint).with_otlp_timeout(Duration::from_secs(60));
    let (guard, handles) = init_providers(&config).unwrap();
    let mut span = handles.tracer.unwrap().start("spike.span");
    span.end();
    handles.meter.unwrap().u64_counter("spike.counter").build().add(1, &[]);

    let deadline = Duration::from_millis(500);
    let started = Instant::now();
    let result = guard.shutdown_blocking(deadline);
    let elapsed = started.elapsed();

    assert!(
        matches!(result, Err(TelemetryError::ShutdownTimeout { deadline: reported }) if reported == deadline),
        "expected a deadline error, got {result:?}"
    );
    assert!(elapsed >= deadline && elapsed < deadline * 3, "{elapsed:?}");
    stub_runtime.shutdown_background();
}

/// SIGTERM while a response is still streaming: the gateway drain gives up,
/// `main` stops the runtime (dropping the connection task and its body), and
/// only then flushes telemetry — so the abandoned request's span and its
/// final measurements are in the export, and the gauge returns to zero.
#[test]
fn requests_abandoned_at_runtime_shutdown_are_finalized_before_telemetry_shutdown() {
    use axum::Router;
    use axum::body::Body;
    use axum::extract::Request;
    use axum::middleware;
    use axum::response::IntoResponse;
    use axum::routing::get;
    use opentelemetry_proto::tonic::metrics::v1::metric::Data;
    use opentelemetry_proto::tonic::metrics::v1::number_data_point::Value as NumberValue;
    use tower::ServiceExt as _;
    use tracing::instrument::WithSubscriber as _;

    use agentic_server::telemetry::build_subscriber;
    use agentic_server::telemetry::http::{HttpMetrics, track_request};

    // The collector lives on its own runtime so it survives the gateway's.
    let stub_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let (endpoint, stub) = stub_runtime.block_on(OtlpStub::spawn(StubMode::Accept));
    let config = enabled_config(&endpoint).with_otlp_timeout(Duration::from_secs(2));
    let (guard, handles) = init_providers(&config).unwrap();
    let dispatch = tracing::Dispatch::new(build_subscriber(
        handles.tracer,
        tracing_subscriber::EnvFilter::new("info"),
        std::io::sink,
    ));
    let router = Router::new()
        .route(
            "/hang",
            get(|| async {
                Body::from_stream(futures::stream::pending::<Result<bytes::Bytes, std::io::Error>>()).into_response()
            }),
        )
        .layer(middleware::from_fn_with_state(
            HttpMetrics::new(&handles.meter.expect("metrics enabled")),
            track_request,
        ));

    let gateway_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (headers_sent, headers_received) = std::sync::mpsc::channel();
    gateway_runtime.spawn(async move {
        let request = Request::builder().uri("/hang").body(Body::empty()).unwrap();
        let response = router.oneshot(request).with_subscriber(dispatch).await.unwrap();
        headers_sent.send(response.status()).unwrap();
        // Hold the never-ending body until the runtime drops this task.
        let _response = response;
        std::future::pending::<()>().await;
    });
    assert_eq!(headers_received.recv_timeout(Duration::from_secs(5)).unwrap(), 200);

    // `main`'s order: runtime first, telemetry second.
    gateway_runtime.shutdown_timeout(Duration::from_secs(1));
    guard.shutdown_blocking(Duration::from_secs(5)).unwrap();

    let traces = stub_runtime.block_on(stub.trace_exports());
    let names: Vec<&str> = traces
        .iter()
        .flat_map(|export| export.resource_spans.iter())
        .flat_map(|resource| resource.scope_spans.iter())
        .flat_map(|scope| scope.spans.iter().map(|span| span.name.as_str()))
        .collect();
    assert_eq!(names, ["GET /hang"], "the abandoned request's span is exported");

    let metrics = stub_runtime.block_on(stub.metric_exports());
    let latest = metrics.last().expect("metrics exported on shutdown");
    let mut active = None;
    let mut durations = 0;
    for metric in latest
        .resource_metrics
        .iter()
        .flat_map(|resource| resource.scope_metrics.iter())
        .flat_map(|scope| scope.metrics.iter())
    {
        match (metric.name.as_str(), metric.data.as_ref()) {
            ("http.server.active_requests", Some(Data::Sum(sum))) => {
                active = Some(
                    sum.data_points
                        .iter()
                        .map(|point| match point.value {
                            Some(NumberValue::AsInt(value)) => value,
                            other => panic!("unexpected value {other:?}"),
                        })
                        .sum::<i64>(),
                );
            }
            ("http.server.request.duration", Some(Data::Histogram(histogram))) => {
                durations += histogram.data_points.iter().map(|point| point.count).sum::<u64>();
            }
            _ => {}
        }
    }
    assert_eq!(active, Some(0), "gauge balanced by the dropped body");
    assert_eq!(durations, 1, "duration recorded for the abandoned request");
    stub_runtime.shutdown_background();
}
