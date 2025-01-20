use backend::{
    kernels::{matmul, KernelBuff, KernelInput},
    VultenInstance, DT_FLOAT,
};
use criterion::{BenchmarkId, Criterion};

pub fn bench(c: &mut Criterion, inst: &VultenInstance) {
    bench_square(c, inst);
    bench_tall(c, inst);
}

fn bench_square(c: &mut Criterion, inst: &VultenInstance) {
    let mut group = c.benchmark_group("MatMul Square");
    group.noise_threshold(0.05);
    let shapes = vec![
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ];
    for shape in shapes {
        let dims: [i64; 2] = [shape.0, shape.1];
        // (m, p) * (p, n)
        //flops = mn(2p - 1)
        let flops = dims[0] * dims[0] * ((2 * dims[0]) - 1);
        group.throughput(criterion::Throughput::Elements(flops as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(shape.0),
            &shape,
            |bench, &_shape| {
                bench.iter_batched(
                    || {
                        let buff_size = dims.iter().product::<i64>() * 4;

                        let staging_buff = inst.create_buffer(
                            backend::memory::VultenBufferType::Staging,
                            buff_size as u64,
                            true,
                            false,
                        );
                        let staging = unsafe {
                            std::slice::from_raw_parts_mut(
                                staging_buff.get_mapped_ptr().unwrap() as *mut f32,
                                buff_size as usize / 4,
                            )
                        };
                        for (idx, val) in staging.iter_mut().enumerate() {
                            *val = idx as f32;
                        }
                        let a_buff = inst.create_buffer(
                            backend::memory::VultenBufferType::Device,
                            buff_size as u64,
                            false,
                            true,
                        );
                        let a = KernelInput {
                            buff: KernelBuff::Buff(a_buff.into()),
                            dims: &dims,
                        };
                        let b_buff = inst.create_buffer(
                            backend::memory::VultenBufferType::Device,
                            buff_size as u64,
                            false,
                            true,
                        );
                        let b = KernelInput {
                            buff: KernelBuff::Buff(b_buff.into()),
                            dims: &dims,
                        };
                        let out_buff = inst.create_buffer(
                            backend::memory::VultenBufferType::Device,
                            buff_size as u64,
                            true,
                            false,
                        );
                        let output = KernelInput {
                            buff: KernelBuff::Buff(out_buff.into()),
                            dims: &dims,
                        };

                        matmul::matmul_inline_transpose::run(
                            inst, DT_FLOAT, &a, false, &b, false, &output,
                        )
                        .unwrap();

                        (a, b, output)
                    },
                    |(a, b, output)| {
                        matmul::matmul_inline_transpose::run(
                            inst, DT_FLOAT, &a, false, &b, false, &output,
                        )
                        .unwrap();
                    },
                    criterion::BatchSize::PerIteration,
                )
            },
        );
    }
}

fn bench_tall(c: &mut Criterion, inst: &VultenInstance) {
    let mut group = c.benchmark_group("MatMul Tall");
    group.noise_threshold(0.05);
    let shapes = vec![
        (32, 8),
        (64, 16),
        (128, 32),
        (256, 64),
        (512, 128),
        (1024, 256),
        (2048, 512),
        (4096, 1024),
        (8192, 2048),
    ];
    for shape in shapes {
        let a_dims: [i64; 2] = [shape.0, shape.1];
        let b_dims: [i64; 2] = [shape.1, shape.0];
        let c_dims: [i64; 2] = [shape.0, shape.0];

        //group.throughput(criterion::Throughput::BytesDecimal(buff_size as u64));
        // (m, p) * (p, n)
        //flops = mn(2p - 1)
        let flops = a_dims[0] * b_dims[1] * ((2 * a_dims[1]) - 1);
        group.throughput(criterion::Throughput::Elements(flops as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(shape.0),
            &shape,
            |bench, &_shape| {
                bench.iter_batched(
                    || {
                        let ab_buff_size = a_dims.iter().product::<i64>() * 4;
                        let c_buff_size = c_dims.iter().product::<i64>() * 4;

                        let staging_buff = inst.create_buffer(
                            backend::memory::VultenBufferType::Staging,
                            ab_buff_size as u64,
                            true,
                            false,
                        );
                        let staging = unsafe {
                            std::slice::from_raw_parts_mut(
                                staging_buff.get_mapped_ptr().unwrap() as *mut f32,
                                ab_buff_size as usize / 4,
                            )
                        };
                        for (idx, val) in staging.iter_mut().enumerate() {
                            *val = idx as f32;
                        }
                        let a_buff = inst.create_buffer(
                            backend::memory::VultenBufferType::Device,
                            ab_buff_size as u64,
                            false,
                            true,
                        );
                        let a = KernelInput {
                            buff: KernelBuff::Buff(a_buff.into()),
                            dims: &a_dims,
                        };
                        let b_buff = inst.create_buffer(
                            backend::memory::VultenBufferType::Device,
                            ab_buff_size as u64,
                            false,
                            true,
                        );
                        let b = KernelInput {
                            buff: KernelBuff::Buff(b_buff.into()),
                            dims: &b_dims,
                        };
                        let out_buff = inst.create_buffer(
                            backend::memory::VultenBufferType::Device,
                            c_buff_size as u64,
                            true,
                            false,
                        );
                        let output = KernelInput {
                            buff: KernelBuff::Buff(out_buff.into()),
                            dims: &c_dims,
                        };

                        matmul::matmul_inline_transpose::run(
                            inst, DT_FLOAT, &a, false, &b, false, &output,
                        )
                        .unwrap();
                        (a, b, output)
                    },
                    |(a, b, output)| {
                        matmul::matmul_inline_transpose::run(
                            inst, DT_FLOAT, &a, false, &b, false, &output,
                        )
                        .unwrap();
                    },
                    criterion::BatchSize::PerIteration,
                )
            },
        );
    }
}
