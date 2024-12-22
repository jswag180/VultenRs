use criterion::{black_box, criterion_group, criterion_main, Criterion};

mod bench_va;

fn benchmark(c: &mut Criterion) {
    //TODO use env var for device selection
    //let inst = backend::VultenInstance::new(Some(1));

    bench_va::bench(c);

    //c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
