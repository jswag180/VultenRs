use criterion::{criterion_group, criterion_main, Criterion};

mod bench_matmul;
mod bench_va;

fn benchmark(c: &mut Criterion) {
    let dev_num: Option<usize> = match std::env::var("VULTEN_BENCH_DEV") {
        Ok(val) => Some(val.parse().unwrap_or_default()),
        Err(_) => None,
    };
    let inst = backend::VultenInstance::new(dev_num);
    println!("Using device: {:}", inst.get_device_name());

    bench_va::bench(c);
    bench_matmul::bench(c, &inst);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
