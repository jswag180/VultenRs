use std::sync::Arc;

use backend::va::{Va, VaAddress};
use criterion::{black_box, BenchmarkId, Criterion};
use rand::Rng;

type PlaceHolderPayload = Arc<u64>;

pub fn bench(c: &mut Criterion) {
    clean(c);
    dirty(c);
}

fn clean(c: &mut Criterion) {
    c.bench_function("Va Clean Alloc", |b| {
        b.iter(|| {
            let va: Va<PlaceHolderPayload> = Va::new();
            for i in 0..1024 {
                let alloc = va.alloc(0, i.into(), 1024).unwrap();
                black_box(alloc);
            }
        })
    });

    let mut group = c.benchmark_group("Va Clean Find");
    group.throughput(criterion::Throughput::Elements(8));
    for total_allocs in (1024..(1024 * 9)).step_by(1024) {
        group.bench_with_input(
            BenchmarkId::from_parameter(total_allocs),
            &total_allocs,
            |b, &total_allocs| {
                b.iter_batched(
                    || {
                        let va: Va<PlaceHolderPayload> = Va::new();
                        let mut allocs: Vec<VaAddress> = Vec::new();
                        let mut rng = rand::thread_rng();
                        let max_size = 4294967296;

                        for i in 0..total_allocs {
                            let alloc = va.alloc(0, i.into(), rng.gen_range(1..max_size)).unwrap();
                            allocs.push(alloc);
                        }

                        let mut to_lookup: Vec<VaAddress> = Vec::new();
                        for _ in 0..8 {
                            let alloc = allocs.remove(rng.gen_range(0..allocs.len()));
                            to_lookup.push(alloc);
                        }

                        (va, to_lookup)
                    },
                    |(va, allocs)| {
                        for alloc_addr in allocs {
                            let alloc = va.find_va(alloc_addr).unwrap();
                            black_box(alloc);
                        }
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn dirty(c: &mut Criterion) {
    let max_size = 4294967296;

    c.bench_function("Dirty Alloc", |b| {
        b.iter_batched(
            || {
                let va: Va<PlaceHolderPayload> = Va::new();
                let mut allocs: Vec<VaAddress> = Vec::new();
                let mut rng = rand::thread_rng();

                for i in 0..1024 {
                    let alloc = va.alloc(0, i.into(), rng.gen_range(1..max_size)).unwrap();
                    allocs.push(alloc);
                }

                for _ in 0..100 {
                    for _ in 0..32 {
                        let alloc = allocs.remove(rng.gen_range(0..allocs.len()));
                        va.free(alloc).unwrap();
                    }

                    for i in 0..32 {
                        let alloc = va.alloc(0, i.into(), rng.gen_range(1..max_size)).unwrap();
                        allocs.push(alloc);
                    }
                }

                va
            },
            |va| {
                let mut rng = rand::thread_rng();
                for i in 0..1024 {
                    let alloc = va.alloc(0, i.into(), rng.gen_range(1..max_size)).unwrap();
                    black_box(alloc);
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    let mut group = c.benchmark_group("Va Dirty Find");
    group.throughput(criterion::Throughput::Elements(8));
    for total_allocs in (1024..(1024 * 9)).step_by(1024) {
        group.bench_with_input(
            BenchmarkId::from_parameter(total_allocs),
            &total_allocs,
            |b, &total_allocs| {
                b.iter_batched(
                    || {
                        let va: Va<PlaceHolderPayload> = Va::new();
                        let mut allocs: Vec<VaAddress> = Vec::new();
                        let mut rng = rand::thread_rng();
                        let max_size = 4294967296;

                        for i in 0..total_allocs {
                            let alloc = va.alloc(0, i.into(), rng.gen_range(1..max_size)).unwrap();
                            allocs.push(alloc);
                        }

                        for _ in 0..100 {
                            for _ in 0..64 {
                                let alloc = allocs.remove(rng.gen_range(0..allocs.len()));
                                va.free(alloc).unwrap();
                            }

                            for i in 0..64 {
                                let alloc =
                                    va.alloc(0, i.into(), rng.gen_range(1..max_size)).unwrap();
                                allocs.push(alloc);
                            }
                        }

                        let mut to_lookup: Vec<VaAddress> = Vec::new();
                        for _ in 0..8 {
                            let alloc = allocs.remove(rng.gen_range(0..allocs.len()));
                            to_lookup.push(alloc);
                        }

                        (va, to_lookup)
                    },
                    |(va, allocs)| {
                        for alloc_addr in allocs {
                            let alloc = va.find_va(alloc_addr).unwrap();
                            black_box(alloc);
                        }
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}
