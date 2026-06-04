mod executor_throughput;
mod storage_crud;

use criterion::criterion_main;

criterion_main!(storage_crud::storage_benches, executor_throughput::executor_benches);
