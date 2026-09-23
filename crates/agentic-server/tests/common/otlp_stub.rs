//! In-process OTLP/HTTP receiver for telemetry tests: records protobuf
//! export bodies (inflating gzip-encoded ones, as a collector would), or
//! holds requests open to simulate a hung collector.

#![allow(dead_code)]

use std::io::Read as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::routing::post;
use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::tonic::common::v1::any_value::Value as AnyValue;
use opentelemetry_proto::tonic::resource::v1::Resource;
use prost::Message as _;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

/// Collector behaviour for one stub instance.
#[derive(Clone, Copy)]
pub enum StubMode {
    /// Accept and acknowledge every export.
    Accept,
    /// Hold every request open for far longer than any test deadline.
    Hang,
}

#[derive(Clone)]
pub struct OtlpStub {
    mode: StubMode,
    pub connections: Arc<AtomicUsize>,
    /// Requests that arrived with `content-encoding: gzip`.
    pub gzip_requests: Arc<AtomicUsize>,
    traces: Arc<Mutex<Vec<Bytes>>>,
    metrics: Arc<Mutex<Vec<Bytes>>>,
}

impl OtlpStub {
    /// Bind on a random port and return the base endpoint plus the handle.
    pub async fn spawn(mode: StubMode) -> (String, Self) {
        let stub = Self {
            mode,
            connections: Arc::new(AtomicUsize::new(0)),
            gzip_requests: Arc::new(AtomicUsize::new(0)),
            traces: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(Vec::new())),
        };
        let app = Router::new()
            .route(
                "/v1/traces",
                post(|state, headers, body| receive(state, headers, "traces", body)),
            )
            .route(
                "/v1/metrics",
                post(|state, headers, body| receive(state, headers, "metrics", body)),
            )
            .with_state(stub.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), stub)
    }

    pub async fn trace_exports(&self) -> Vec<ExportTraceServiceRequest> {
        self.traces
            .lock()
            .await
            .iter()
            .map(|body| ExportTraceServiceRequest::decode(body.clone()).unwrap())
            .collect()
    }

    pub async fn metric_exports(&self) -> Vec<ExportMetricsServiceRequest> {
        self.metrics
            .lock()
            .await
            .iter()
            .map(|body| ExportMetricsServiceRequest::decode(body.clone()).unwrap())
            .collect()
    }
}

async fn receive(State(stub): State<OtlpStub>, headers: HeaderMap, path: &'static str, body: Bytes) -> StatusCode {
    stub.connections.fetch_add(1, Ordering::SeqCst);
    assert_eq!(
        headers.get(header::CONTENT_TYPE).and_then(|value| value.to_str().ok()),
        Some("application/x-protobuf"),
        "OTLP/HTTP protobuf is the only wire format"
    );
    match stub.mode {
        StubMode::Hang => {
            tokio::time::sleep(Duration::from_secs(120)).await;
            StatusCode::OK
        }
        StubMode::Accept => {
            let body = match headers.get(header::CONTENT_ENCODING).map(HeaderValue::as_bytes) {
                None => body,
                Some(b"gzip") => {
                    stub.gzip_requests.fetch_add(1, Ordering::SeqCst);
                    let mut inflated = Vec::new();
                    flate2::read::GzDecoder::new(&body[..])
                        .read_to_end(&mut inflated)
                        .expect("gzip body inflates");
                    Bytes::from(inflated)
                }
                Some(other) => panic!("unexpected content-encoding {}", String::from_utf8_lossy(other)),
            };
            let store = if path == "traces" { &stub.traces } else { &stub.metrics };
            store.lock().await.push(body);
            StatusCode::OK
        }
    }
}

/// `service.name` from an exported resource.
pub fn service_name(resource: Option<&Resource>) -> Option<&str> {
    resource?
        .attributes
        .iter()
        .find(|attribute| attribute.key == "service.name")
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|value| match &value.value {
            Some(AnyValue::StringValue(name)) => Some(name.as_str()),
            _ => None,
        })
}
