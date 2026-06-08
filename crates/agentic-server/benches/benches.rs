mod gateway_bench;
mod proxy_bench;

use criterion::criterion_main;

criterion_main!(proxy_bench::proxy_benches, gateway_bench::gateway_benches);
