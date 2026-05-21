// Storage CRUD operations benchmark
// Run with: cargo bench --bench storage_crud
// Or with custom iterations: cargo bench --bench storage_crud -- --iterations 100

use std::env;
use std::sync::Arc;
use std::time::Instant;

use agentic_api::storage::{ConversationStore, InOutItem, ResponseMetadata, ResponseStore, SchemaManager};
use agentic_api::types::io::{InputItem, InputMessage, InputMessageContent, OutputItem, OutputMessage};

const DEFAULT_ITERATIONS: usize = 50;

#[derive(Debug)]
struct BenchmarkResult {
    operation: String,
    iterations: usize,
    total_duration_ms: f64,
    average_duration_ms: f64,
    min_duration_ms: f64,
    max_duration_ms: f64,
    throughput_ops_per_sec: f64,
}

impl BenchmarkResult {
    fn new(operation: &str, iterations: usize, durations: Vec<f64>) -> Self {
        let total = durations.iter().sum::<f64>();
        let avg = total / iterations as f64;
        let min = durations.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = durations.iter().cloned().fold(0.0, f64::max);
        let throughput = 1000.0 / avg;

        Self {
            operation: operation.to_string(),
            iterations,
            total_duration_ms: total,
            average_duration_ms: avg,
            min_duration_ms: min,
            max_duration_ms: max,
            throughput_ops_per_sec: throughput,
        }
    }

    fn print(&self) {
        println!("\n=== Benchmark: {} ===", self.operation);
        println!("Iterations: {}", self.iterations);
        println!("Total Duration: {:.2}ms", self.total_duration_ms);
        println!("Average: {:.4}ms/op", self.average_duration_ms);
        println!("Min: {:.4}ms", self.min_duration_ms);
        println!("Max: {:.4}ms", self.max_duration_ms);
        println!("Throughput: {:.2} ops/sec", self.throughput_ops_per_sec);
    }
}

fn create_test_items() -> Vec<InOutItem> {
    let input_item = InputItem::Message(InputMessage {
        role: "user".to_string(),
        content: InputMessageContent::Text("Test message".to_string()),
    });

    let output_msg = OutputMessage::new("msg_123", "completed");

    vec![
        InOutItem::Input(input_item.clone()),
        InOutItem::Output(OutputItem::Message(output_msg)),
        InOutItem::Input(input_item),
    ]
}

fn create_test_metadata() -> ResponseMetadata {
    ResponseMetadata::default()
}

fn parse_iterations() -> usize {
    let args: Vec<String> = env::args().collect();

    for (i, arg) in args.iter().enumerate() {
        if arg == "--iterations" || arg == "-i" {
            if let Some(next_arg) = args.get(i + 1) {
                if let Ok(iterations) = next_arg.parse::<usize>() {
                    if iterations > 0 {
                        return iterations;
                    } else {
                        eprintln!("Warning: iterations must be > 0, using default: {}", DEFAULT_ITERATIONS);
                        return DEFAULT_ITERATIONS;
                    }
                } else {
                    eprintln!(
                        "Warning: could not parse iterations value '{}', using default: {}",
                        next_arg, DEFAULT_ITERATIONS
                    );
                    return DEFAULT_ITERATIONS;
                }
            }
        }
    }

    DEFAULT_ITERATIONS
}

fn print_usage() {
    println!("\n=== Benchmark Usage ===");
    println!("Default (50 iterations):");
    println!("  cargo bench --bench storage_crud\n");
    println!("Custom iterations:");
    println!("  cargo bench --bench storage_crud -- --iterations 100");
    println!("  cargo bench --bench storage_crud -- -i 200\n");
}

async fn create_test_pool() -> Arc<sqlx::Pool<sqlx::Any>> {
    sqlx::any::install_default_drivers();
    let pool = sqlx::any::AnyPoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .expect("failed to create test pool");
    let pool = Arc::new(pool);

    let schema_manager = SchemaManager::new_for_test(pool.as_ref());
    schema_manager
        .ensure_ready()
        .await
        .expect("failed to initialize schema");

    pool
}

#[tokio::main]
async fn main() {
    let iterations = parse_iterations();

    println!("Starting Storage CRUD Benchmarks...");
    println!("Iterations: {}\n", iterations);

    bench_conversation_persist(iterations).await;
    bench_conversation_rehydrate(iterations).await;
    bench_response_persist(iterations).await;
    bench_response_rehydrate(iterations).await;

    print_usage();
    println!("\n✅ All benchmarks completed");
}

async fn bench_conversation_persist(iterations: usize) {
    let pool = create_test_pool().await;

    let store = ConversationStore::new(Arc::clone(&pool));

    let conversation = match store.create().await {
        Ok(conv) => conv,
        Err(e) => {
            eprintln!("Failed to create conversation: {}", e);
            return;
        }
    };
    let conversation_id = &conversation.conversation_id;

    let test_items_list: Vec<Vec<InOutItem>> = (0..100).map(|_| create_test_items()).collect();
    let test_metadata = create_test_metadata();
    let mut durations = Vec::new();
    let mut previous_response_id: Option<String> = None;

    for i in 0..iterations {
        let new_items = test_items_list[i % 100].clone();
        let response_id = format!("resp_{}", i);

        let start = Instant::now();
        let result = store
            .persist(
                conversation_id,
                &response_id,
                previous_response_id.as_deref(),
                new_items,
                &test_metadata,
            )
            .await;
        let duration = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(_) => {
                previous_response_id = Some(response_id.clone());
                durations.push(duration);
            }
            Err(e) => {
                eprintln!("Persist operation failed: {}", e);
                return;
            }
        }
    }

    let result = BenchmarkResult::new("ConversationStore::persist", iterations, durations);
    result.print();
}

async fn bench_response_persist(iterations: usize) {
    let pool = create_test_pool().await;

    let store = ResponseStore::new(Arc::clone(&pool));

    let test_items_list: Vec<Vec<InOutItem>> = (0..100).map(|_| create_test_items()).collect();
    let test_metadata = create_test_metadata();
    let mut durations = Vec::new();
    let mut previous_response_id: Option<String> = None;

    for i in 0..iterations {
        let new_items = test_items_list[i % 100].clone();
        let response_id = format!("resp_{}", i);

        let start = Instant::now();
        let result = store
            .persist(&response_id, previous_response_id.as_deref(), new_items, &test_metadata)
            .await;
        let duration = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(_) => {
                previous_response_id = Some(response_id);
                durations.push(duration);
            }
            Err(e) => {
                eprintln!("Persist operation failed: {}", e);
                return;
            }
        }
    }

    let result = BenchmarkResult::new("ResponseStore::persist", iterations, durations);
    result.print();
}

async fn bench_conversation_rehydrate(iterations: usize) {
    let pool = create_test_pool().await;

    let store = ConversationStore::new(Arc::clone(&pool));

    let conversation = match store.create().await {
        Ok(conv) => conv,
        Err(e) => {
            eprintln!("Failed to create conversation: {}", e);
            return;
        }
    };
    let conversation_id = &conversation.conversation_id;

    let test_items_list: Vec<Vec<InOutItem>> = (0..100).map(|_| create_test_items()).collect();
    let test_metadata = create_test_metadata();

    let mut previous_response_id: Option<String> = None;
    for i in 0..iterations {
        let new_items = test_items_list[i % 100].clone();
        let response_id = format!("resp_{}", i);

        let _ = store
            .persist(
                conversation_id,
                &response_id,
                previous_response_id.as_deref(),
                new_items,
                &test_metadata,
            )
            .await;

        previous_response_id = Some(response_id);
    }

    // Now benchmark rehydrate
    let mut durations = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        let result = store.rehydrate(conversation_id).await;
        let duration = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(_) => {
                durations.push(duration);
            }
            Err(e) => {
                eprintln!("Rehydrate operation failed: {}", e);
                return;
            }
        }
    }

    let result = BenchmarkResult::new("ConversationStore::rehydrate", iterations, durations);
    result.print();
}

async fn bench_response_rehydrate(iterations: usize) {
    let pool = create_test_pool().await;

    let store = ResponseStore::new(Arc::clone(&pool));

    let test_items_list: Vec<Vec<InOutItem>> = (0..100).map(|_| create_test_items()).collect();
    let test_metadata = create_test_metadata();

    // Populate with data first, chaining responses
    let mut response_ids = Vec::new();
    let mut previous_response_id: Option<String> = None;

    for i in 0..iterations {
        let new_items = test_items_list[i % 100].clone();
        let response_id = format!("resp_{}", i);

        if let Ok(_) = store
            .persist(&response_id, previous_response_id.as_deref(), new_items, &test_metadata)
            .await
        {
            response_ids.push(response_id.clone());
            previous_response_id = Some(response_id);
        }
    }

    // Fetch the response data for rehydration
    let mut response_data_list = Vec::new();
    for response_id in &response_ids {
        if let Ok(Some(resp_data)) = store.get(response_id).await {
            response_data_list.push(resp_data);
        }
    }

    // Now benchmark rehydrate
    let mut durations = Vec::new();

    for response_data in &response_data_list {
        let start = Instant::now();
        let result = store.rehydrate(response_data).await;
        let duration = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(_) => {
                durations.push(duration);
            }
            Err(e) => {
                eprintln!("Rehydrate operation failed: {}", e);
                return;
            }
        }
    }

    let result = BenchmarkResult::new("ResponseStore::rehydrate", iterations, durations);
    result.print();
}
