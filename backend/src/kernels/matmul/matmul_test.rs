use crate::{
    kernels::{matmul::MatMulKernel, KernelBuff},
    memory::{VultenBufferType, VultenCpyInfo},
    test_utills::buff_size_from_dims,
    DT_FLOAT, TEST_INST,
};

struct MatMulTest<'a> {
    a_dims: &'a [i64],
    a_transpose: bool,
    b_dims: &'a [i64],
    b_transpose: bool,
}

impl<'a> MatMulTest<'a> {
    pub fn new(a_dims: &'a [i64], a_transpose: bool, b_dims: &'a [i64], b_transpose: bool) -> Self {
        Self {
            a_dims,
            a_transpose,
            b_dims,
            b_transpose,
        }
    }

    pub fn a(&self) -> Vec<f32> {
        let mut vec = Vec::default();
        for i in 0..self.a_dims.iter().product() {
            vec.push(i as f32);
        }

        vec
    }

    pub fn b(&self) -> Vec<f32> {
        let mut vec = Vec::default();
        for i in (0..self.b_dims.iter().product()).rev() {
            vec.push(i as f32);
        }

        vec
    }

    pub fn c_dims(&self) -> Result<Vec<i64>, ()> {
        if self.a_dims.len() != self.b_dims.len() {
            return Err(());
        }

        let a_post = if self.a_transpose {
            if self.a_dims.len() == 2 {
                vec![self.a_dims[1], self.a_dims[0]]
            } else if self.a_dims.len() == 3 {
                vec![self.a_dims[0], self.a_dims[2], self.a_dims[1]]
            } else {
                return Err(());
            }
        } else {
            self.a_dims.to_vec()
        };

        let b_post = if self.b_transpose {
            if self.b_dims.len() == 2 {
                vec![self.b_dims[1], self.b_dims[0]]
            } else if self.b_dims.len() == 3 {
                vec![self.b_dims[0], self.b_dims[2], self.b_dims[1]]
            } else {
                return Err(());
            }
        } else {
            self.b_dims.to_vec()
        };

        if a_post.len() == 2 {
            if a_post[1] != b_post[0] {
                return Err(());
            }
            return Ok(vec![a_post[0], b_post[1]]);
        } else {
            if a_post[2] != b_post[1] {
                return Err(());
            }
            return Ok(vec![a_post[0].max(b_post[0]), a_post[1], b_post[2]]);
        }
    }

    pub fn c(&self) -> Vec<f32> {
        let a = self.a();
        let b = self.b();
        let mut vec = Vec::default();

        let a_post = if self.a_transpose {
            if self.a_dims.len() == 2 {
                vec![self.a_dims[1], self.a_dims[0]]
            } else if self.a_dims.len() == 3 {
                vec![self.a_dims[0], self.a_dims[2], self.a_dims[1]]
            } else {
                panic!()
            }
        } else {
            self.a_dims.to_vec()
        };
        let a_b = if a_post.len() == 2 { 1 } else { a_post[0] };
        let a_x = if a_post.len() == 2 {
            a_post[0]
        } else {
            a_post[1]
        };
        let a_y = if a_post.len() == 2 {
            a_post[1]
        } else {
            a_post[2]
        };

        let b_post = if self.b_transpose {
            if self.b_dims.len() == 2 {
                vec![self.b_dims[1], self.b_dims[0]]
            } else if self.b_dims.len() == 3 {
                vec![self.b_dims[0], self.b_dims[2], self.b_dims[1]]
            } else {
                panic!()
            }
        } else {
            self.b_dims.to_vec()
        };
        let b_b = if b_post.len() == 2 { 1 } else { b_post[0] };
        let b_x = if b_post.len() == 2 {
            b_post[0]
        } else {
            b_post[1]
        };
        let b_y = if b_post.len() == 2 {
            b_post[1]
        } else {
            b_post[2]
        };

        for batch in 0..(a_b.max(b_b)) {
            for x in 0..a_x {
                for y in 0..b_y {
                    let mut dot = 0.0;
                    for i in 0..a_y {
                        let a_idx = if self.a_transpose {
                            (batch.min(a_b - 1) * a_x * a_y) + (i * a_x) + x
                        } else {
                            (batch.min(a_b - 1) * a_x * a_y) + (x * a_y) + i
                        };
                        let b_idx = if self.b_transpose {
                            (batch.min(b_b - 1) * b_x * b_y) + (y * b_x) + i
                        } else {
                            (batch.min(b_b - 1) * b_x * b_y) + (i * b_y) + y
                        };
                        dot += a[a_idx as usize] * b[b_idx as usize];
                    }

                    vec.push(dot);
                }
            }
        }

        vec
    }
}

#[test]
fn one_n_one() {
    let a_dims = vec![1, 4];
    let a_transpose = false;
    let b_dims = vec![4, 1];
    let b_transpose = false;
    let test = MatMulTest::new(&a_dims, a_transpose, &b_dims, b_transpose);
    let c_dims = test.c_dims().unwrap();

    let a_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&a_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.a(), &a_buff, 0, None)
        .unwrap();
    let a = KernelBuff::Buff(a_buff.into());

    let b_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&b_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.b(), &b_buff, 0, None)
        .unwrap();
    let b = KernelBuff::Buff(b_buff.into());

    let c_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&c_dims, DT_FLOAT),
        true,
        false,
    );
    let c = KernelBuff::Buff(c_buff.into());

    MatMulKernel::new(&TEST_INST, DT_FLOAT)
        .a(a.clone(), &a_dims, a_transpose)
        .unwrap()
        .b(b.clone(), &b_dims, b_transpose)
        .unwrap()
        .output(c.clone(), &c_dims)
        .unwrap()
        .build(None)
        .unwrap()
        .run()
        .unwrap();

    let buff_size = buff_size_from_dims(&c_dims, DT_FLOAT);
    let host_buff = TEST_INST.create_buffer(VultenBufferType::Host, buff_size, true, true);
    let cpy_info = VultenCpyInfo::default().size(buff_size as u64);
    TEST_INST.blocking_cpy(
        c.get_buffer().unwrap().0.vk_buffer,
        host_buff.vk_buffer,
        cpy_info,
    );
    let result = unsafe {
        std::slice::from_raw_parts_mut(
            host_buff.get_mapped_ptr().unwrap() as *mut f32,
            buff_size as usize,
        )
    };

    for (res, truth) in result.iter().zip(test.c()) {
        assert_eq!(*res, truth);
    }
}

#[test]
fn even_square() {
    let a_dims = vec![8, 8];
    let a_transpose = false;
    let b_dims = vec![8, 8];
    let b_transpose = false;
    let test = MatMulTest::new(&a_dims, a_transpose, &b_dims, b_transpose);
    let c_dims = test.c_dims().unwrap();

    let a_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&a_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.a(), &a_buff, 0, None)
        .unwrap();
    let a = KernelBuff::Buff(a_buff.into());

    let b_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&b_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.b(), &b_buff, 0, None)
        .unwrap();
    let b = KernelBuff::Buff(b_buff.into());

    let c_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&c_dims, DT_FLOAT),
        true,
        false,
    );
    let c = KernelBuff::Buff(c_buff.into());

    MatMulKernel::new(&TEST_INST, DT_FLOAT)
        .a(a.clone(), &a_dims, a_transpose)
        .unwrap()
        .b(b.clone(), &b_dims, b_transpose)
        .unwrap()
        .output(c.clone(), &c_dims)
        .unwrap()
        .build(None)
        .unwrap()
        .run()
        .unwrap();

    let buff_size = buff_size_from_dims(&c_dims, DT_FLOAT);
    let host_buff = TEST_INST.create_buffer(VultenBufferType::Host, buff_size, true, true);
    let cpy_info = VultenCpyInfo::default().size(buff_size as u64);
    TEST_INST.blocking_cpy(
        c.get_buffer().unwrap().0.vk_buffer,
        host_buff.vk_buffer,
        cpy_info,
    );
    let result = unsafe {
        std::slice::from_raw_parts_mut(
            host_buff.get_mapped_ptr().unwrap() as *mut f32,
            buff_size as usize,
        )
    };

    for (res, truth) in result.iter().zip(test.c()) {
        assert_eq!(*res, truth);
    }
}

#[test]
fn odd_square() {
    let a_dims = vec![7, 7];
    let a_transpose = false;
    let b_dims = vec![7, 7];
    let b_transpose = false;
    let test = MatMulTest::new(&a_dims, a_transpose, &b_dims, b_transpose);
    let c_dims = test.c_dims().unwrap();

    let a_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&a_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.a(), &a_buff, 0, None)
        .unwrap();
    let a = KernelBuff::Buff(a_buff.into());

    let b_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&b_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.b(), &b_buff, 0, None)
        .unwrap();
    let b = KernelBuff::Buff(b_buff.into());

    let c_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&c_dims, DT_FLOAT),
        true,
        false,
    );
    let c = KernelBuff::Buff(c_buff.into());

    MatMulKernel::new(&TEST_INST, DT_FLOAT)
        .a(a.clone(), &a_dims, a_transpose)
        .unwrap()
        .b(b.clone(), &b_dims, b_transpose)
        .unwrap()
        .output(c.clone(), &c_dims)
        .unwrap()
        .build(None)
        .unwrap()
        .run()
        .unwrap();

    let buff_size = buff_size_from_dims(&c_dims, DT_FLOAT);
    let host_buff = TEST_INST.create_buffer(VultenBufferType::Host, buff_size, true, true);
    let cpy_info = VultenCpyInfo::default().size(buff_size as u64);
    TEST_INST.blocking_cpy(
        c.get_buffer().unwrap().0.vk_buffer,
        host_buff.vk_buffer,
        cpy_info,
    );
    let result = unsafe {
        std::slice::from_raw_parts_mut(
            host_buff.get_mapped_ptr().unwrap() as *mut f32,
            buff_size as usize,
        )
    };

    for (res, truth) in result.iter().zip(test.c()) {
        assert_eq!(*res, truth);
    }
}

#[test]
fn transpose_a() {
    let a_dims = vec![8, 7];
    let a_transpose = true;
    let b_dims = vec![8, 6];
    let b_transpose = false;
    let test = MatMulTest::new(&a_dims, a_transpose, &b_dims, b_transpose);
    let c_dims = test.c_dims().unwrap();

    let a_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&a_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.a(), &a_buff, 0, None)
        .unwrap();
    let a = KernelBuff::Buff(a_buff.into());

    let b_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&b_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.b(), &b_buff, 0, None)
        .unwrap();
    let b = KernelBuff::Buff(b_buff.into());

    let c_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&c_dims, DT_FLOAT),
        true,
        false,
    );
    let c = KernelBuff::Buff(c_buff.into());

    MatMulKernel::new(&TEST_INST, DT_FLOAT)
        .a(a.clone(), &a_dims, a_transpose)
        .unwrap()
        .b(b.clone(), &b_dims, b_transpose)
        .unwrap()
        .output(c.clone(), &c_dims)
        .unwrap()
        .build(None)
        .unwrap()
        .run()
        .unwrap();

    let buff_size = buff_size_from_dims(&c_dims, DT_FLOAT);
    let host_buff = TEST_INST.create_buffer(VultenBufferType::Host, buff_size, true, true);
    let cpy_info = VultenCpyInfo::default().size(buff_size as u64);
    TEST_INST.blocking_cpy(
        c.get_buffer().unwrap().0.vk_buffer,
        host_buff.vk_buffer,
        cpy_info,
    );
    let result = unsafe {
        std::slice::from_raw_parts_mut(
            host_buff.get_mapped_ptr().unwrap() as *mut f32,
            buff_size as usize,
        )
    };

    for (res, truth) in result.iter().zip(test.c()) {
        assert_eq!(*res, truth);
    }
}

#[test]
fn transpose_b() {
    let a_dims = vec![7, 8];
    let a_transpose = false;
    let b_dims = vec![6, 8];
    let b_transpose = true;
    let test = MatMulTest::new(&a_dims, a_transpose, &b_dims, b_transpose);
    let c_dims = test.c_dims().unwrap();

    let a_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&a_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.a(), &a_buff, 0, None)
        .unwrap();
    let a = KernelBuff::Buff(a_buff.into());

    let b_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&b_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.b(), &b_buff, 0, None)
        .unwrap();
    let b = KernelBuff::Buff(b_buff.into());

    let c_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&c_dims, DT_FLOAT),
        true,
        false,
    );
    let c = KernelBuff::Buff(c_buff.into());

    MatMulKernel::new(&TEST_INST, DT_FLOAT)
        .a(a.clone(), &a_dims, a_transpose)
        .unwrap()
        .b(b.clone(), &b_dims, b_transpose)
        .unwrap()
        .output(c.clone(), &c_dims)
        .unwrap()
        .build(None)
        .unwrap()
        .run()
        .unwrap();

    let buff_size = buff_size_from_dims(&c_dims, DT_FLOAT);
    let host_buff = TEST_INST.create_buffer(VultenBufferType::Host, buff_size, true, true);
    let cpy_info = VultenCpyInfo::default().size(buff_size as u64);
    TEST_INST.blocking_cpy(
        c.get_buffer().unwrap().0.vk_buffer,
        host_buff.vk_buffer,
        cpy_info,
    );
    let result = unsafe {
        std::slice::from_raw_parts_mut(
            host_buff.get_mapped_ptr().unwrap() as *mut f32,
            buff_size as usize,
        )
    };

    for (res, truth) in result.iter().zip(test.c()) {
        assert_eq!(*res, truth);
    }
}

#[test]
fn broadcast_a() {
    let a_dims = vec![1, 7, 8];
    let a_transpose = false;
    let b_dims = vec![2, 8, 6];
    let b_transpose = false;
    let test = MatMulTest::new(&a_dims, a_transpose, &b_dims, b_transpose);
    let c_dims = test.c_dims().unwrap();

    let a_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&a_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.a(), &a_buff, 0, None)
        .unwrap();
    let a = KernelBuff::Buff(a_buff.into());

    let b_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&b_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.b(), &b_buff, 0, None)
        .unwrap();
    let b = KernelBuff::Buff(b_buff.into());

    let c_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&c_dims, DT_FLOAT),
        true,
        false,
    );
    let c = KernelBuff::Buff(c_buff.into());

    MatMulKernel::new(&TEST_INST, DT_FLOAT)
        .a(a.clone(), &a_dims, a_transpose)
        .unwrap()
        .b(b.clone(), &b_dims, b_transpose)
        .unwrap()
        .output(c.clone(), &c_dims)
        .unwrap()
        .build(None)
        .unwrap()
        .run()
        .unwrap();

    let buff_size = buff_size_from_dims(&c_dims, DT_FLOAT);
    let host_buff = TEST_INST.create_buffer(VultenBufferType::Host, buff_size, true, true);
    let cpy_info = VultenCpyInfo::default().size(buff_size as u64);
    TEST_INST.blocking_cpy(
        c.get_buffer().unwrap().0.vk_buffer,
        host_buff.vk_buffer,
        cpy_info,
    );
    let result = unsafe {
        std::slice::from_raw_parts_mut(
            host_buff.get_mapped_ptr().unwrap() as *mut f32,
            buff_size as usize,
        )
    };

    for (res, truth) in result.iter().zip(test.c()) {
        assert_eq!(*res, truth);
    }
}

#[test]
fn broadcast_a_transpose() {
    let a_dims = vec![1, 7, 8];
    let a_transpose = false;
    let b_dims = vec![2, 6, 8];
    let b_transpose = true;
    let test = MatMulTest::new(&a_dims, a_transpose, &b_dims, b_transpose);
    let c_dims = test.c_dims().unwrap();

    let a_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&a_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.a(), &a_buff, 0, None)
        .unwrap();
    let a = KernelBuff::Buff(a_buff.into());

    let b_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&b_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.b(), &b_buff, 0, None)
        .unwrap();
    let b = KernelBuff::Buff(b_buff.into());

    let c_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&c_dims, DT_FLOAT),
        true,
        false,
    );
    let c = KernelBuff::Buff(c_buff.into());

    MatMulKernel::new(&TEST_INST, DT_FLOAT)
        .a(a.clone(), &a_dims, a_transpose)
        .unwrap()
        .b(b.clone(), &b_dims, b_transpose)
        .unwrap()
        .output(c.clone(), &c_dims)
        .unwrap()
        .build(None)
        .unwrap()
        .run()
        .unwrap();

    let buff_size = buff_size_from_dims(&c_dims, DT_FLOAT);
    let host_buff = TEST_INST.create_buffer(VultenBufferType::Host, buff_size, true, true);
    let cpy_info = VultenCpyInfo::default().size(buff_size as u64);
    TEST_INST.blocking_cpy(
        c.get_buffer().unwrap().0.vk_buffer,
        host_buff.vk_buffer,
        cpy_info,
    );
    let result = unsafe {
        std::slice::from_raw_parts_mut(
            host_buff.get_mapped_ptr().unwrap() as *mut f32,
            buff_size as usize,
        )
    };

    for (res, truth) in result.iter().zip(test.c()) {
        assert_eq!(*res, truth);
    }
}

#[test]
fn broadcast_b() {
    let a_dims = vec![2, 7, 8];
    let a_transpose = false;
    let b_dims = vec![1, 8, 6];
    let b_transpose = false;
    let test = MatMulTest::new(&a_dims, a_transpose, &b_dims, b_transpose);
    let c_dims = test.c_dims().unwrap();

    let a_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&a_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.a(), &a_buff, 0, None)
        .unwrap();
    let a = KernelBuff::Buff(a_buff.into());

    let b_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&b_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.b(), &b_buff, 0, None)
        .unwrap();
    let b = KernelBuff::Buff(b_buff.into());

    let c_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&c_dims, DT_FLOAT),
        true,
        false,
    );
    let c = KernelBuff::Buff(c_buff.into());

    MatMulKernel::new(&TEST_INST, DT_FLOAT)
        .a(a.clone(), &a_dims, a_transpose)
        .unwrap()
        .b(b.clone(), &b_dims, b_transpose)
        .unwrap()
        .output(c.clone(), &c_dims)
        .unwrap()
        .build(None)
        .unwrap()
        .run()
        .unwrap();

    let buff_size = buff_size_from_dims(&c_dims, DT_FLOAT);
    let host_buff = TEST_INST.create_buffer(VultenBufferType::Host, buff_size, true, true);
    let cpy_info = VultenCpyInfo::default().size(buff_size as u64);
    TEST_INST.blocking_cpy(
        c.get_buffer().unwrap().0.vk_buffer,
        host_buff.vk_buffer,
        cpy_info,
    );
    let result = unsafe {
        std::slice::from_raw_parts_mut(
            host_buff.get_mapped_ptr().unwrap() as *mut f32,
            buff_size as usize,
        )
    };

    for (res, truth) in result.iter().zip(test.c()) {
        assert_eq!(*res, truth);
    }
}

#[test]
fn broadcast_b_transpose() {
    let a_dims = vec![2, 7, 8];
    let a_transpose = false;
    let b_dims = vec![1, 6, 8];
    let b_transpose = true;
    let test = MatMulTest::new(&a_dims, a_transpose, &b_dims, b_transpose);
    let c_dims = test.c_dims().unwrap();

    let a_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&a_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.a(), &a_buff, 0, None)
        .unwrap();
    let a = KernelBuff::Buff(a_buff.into());

    let b_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&b_dims, DT_FLOAT),
        false,
        true,
    );
    TEST_INST
        .upload_to_device_buff(&test.b(), &b_buff, 0, None)
        .unwrap();
    let b = KernelBuff::Buff(b_buff.into());

    let c_buff = TEST_INST.create_buffer(
        VultenBufferType::Device,
        buff_size_from_dims(&c_dims, DT_FLOAT),
        true,
        false,
    );
    let c = KernelBuff::Buff(c_buff.into());

    MatMulKernel::new(&TEST_INST, DT_FLOAT)
        .a(a.clone(), &a_dims, a_transpose)
        .unwrap()
        .b(b.clone(), &b_dims, b_transpose)
        .unwrap()
        .output(c.clone(), &c_dims)
        .unwrap()
        .build(None)
        .unwrap()
        .run()
        .unwrap();

    let buff_size = buff_size_from_dims(&c_dims, DT_FLOAT);
    let host_buff = TEST_INST.create_buffer(VultenBufferType::Host, buff_size, true, true);
    let cpy_info = VultenCpyInfo::default().size(buff_size as u64);
    TEST_INST.blocking_cpy(
        c.get_buffer().unwrap().0.vk_buffer,
        host_buff.vk_buffer,
        cpy_info,
    );
    let result = unsafe {
        std::slice::from_raw_parts_mut(
            host_buff.get_mapped_ptr().unwrap() as *mut f32,
            buff_size as usize,
        )
    };

    for (res, truth) in result.iter().zip(test.c()) {
        assert_eq!(*res, truth);
    }
}

#[ignore]
#[test]
fn exhaustive() {
    let max_batch = 3;
    let max_x = 10;
    let max_y = 10;

    for ba in 0..max_batch {
        for bb in 0..max_batch {
            for ax in 1..max_x {
                for ay in 1..max_y {
                    for bx in 1..max_x {
                        for by in 1..max_y {
                            for ta in 0..2 {
                                for tb in 0..2 {
                                    let a_dims = if ba != 0 || bb != 0 {
                                        vec![ba.max(1), ax, ay]
                                    } else {
                                        vec![ax, ay]
                                    };
                                    let a_transpose = if ta == 0 { false } else { true };
                                    let b_dims = if ba != 0 || bb != 0 {
                                        vec![bb.max(1), bx, by]
                                    } else {
                                        vec![bx, by]
                                    };
                                    let b_transpose = if tb == 0 { false } else { true };
                                    let test =
                                        MatMulTest::new(&a_dims, a_transpose, &b_dims, b_transpose);
                                    let c_dims = match test.c_dims() {
                                        Ok(val) => val,
                                        Err(_) => break,
                                    };
                                    println!("A: {:?} Transpose: {:?}", a_dims, a_transpose);
                                    println!("B: {:?} Transpose: {:?}", b_dims, b_transpose);
                                    println!("C: {:?}", a_dims);

                                    let a_buff = TEST_INST.create_buffer(
                                        VultenBufferType::Device,
                                        buff_size_from_dims(&a_dims, DT_FLOAT),
                                        false,
                                        true,
                                    );
                                    TEST_INST
                                        .upload_to_device_buff(&test.a(), &a_buff, 0, None)
                                        .unwrap();
                                    let a = KernelBuff::Buff(a_buff.into());

                                    let b_buff = TEST_INST.create_buffer(
                                        VultenBufferType::Device,
                                        buff_size_from_dims(&b_dims, DT_FLOAT),
                                        false,
                                        true,
                                    );
                                    TEST_INST
                                        .upload_to_device_buff(&test.b(), &b_buff, 0, None)
                                        .unwrap();
                                    let b = KernelBuff::Buff(b_buff.into());

                                    let c_buff = TEST_INST.create_buffer(
                                        VultenBufferType::Device,
                                        buff_size_from_dims(&c_dims, DT_FLOAT),
                                        true,
                                        false,
                                    );
                                    let c = KernelBuff::Buff(c_buff.into());

                                    MatMulKernel::new(&TEST_INST, DT_FLOAT)
                                        .a(a.clone(), &a_dims, a_transpose)
                                        .unwrap()
                                        .b(b.clone(), &b_dims, b_transpose)
                                        .unwrap()
                                        .output(c.clone(), &c_dims)
                                        .unwrap()
                                        .build(None)
                                        .unwrap()
                                        .run()
                                        .unwrap();

                                    let buff_size = buff_size_from_dims(&c_dims, DT_FLOAT);
                                    let host_buff = TEST_INST.create_buffer(
                                        VultenBufferType::Host,
                                        buff_size,
                                        true,
                                        true,
                                    );
                                    let cpy_info = VultenCpyInfo::default().size(buff_size as u64);
                                    TEST_INST.blocking_cpy(
                                        c.get_buffer().unwrap().0.vk_buffer,
                                        host_buff.vk_buffer,
                                        cpy_info,
                                    );
                                    let result = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            host_buff.get_mapped_ptr().unwrap() as *mut f32,
                                            buff_size as usize,
                                        )
                                    };

                                    for (res, truth) in result.iter().zip(test.c()) {
                                        assert_eq!(*res, truth);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
