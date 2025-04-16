use ash::vk::{
    self, DescriptorType, PipelineBindPoint, PushConstantRange, QueueFlags, ShaderStageFlags, SpecializationInfo, SpecializationMapEntry, SubmitInfo, WriteDescriptorSet
};
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    cmd_buff::CommandBufferBuilder,
    compiler,
    descriptor::VultenDescriptor,
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

use super::{KernelBuff, MatMulKernel, MatMulKernelVersion};

pub const MATMUL_SOURCE: &str = include_str!("matmul_shared.comp");

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct MatmulSharedPipelineSpec {
    local_x: u32,
    local_y: u32,
    n: u32,
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct MatmulSharedPushConst {
    asd: u32
}

impl PushConstSpec for MatmulSharedPushConst {
    fn get_ranges() -> &'static [PushConstantRange] {
        &[PushConstantRange {
            offset: 0,
            stage_flags: ShaderStageFlags::COMPUTE,
            size: std::mem::size_of::<Self>() as u32,
        }]
    }

    #[inline]
    fn get_slice(&self) -> &[u8] {
        let slice: &[u8; 4] = zerocopy::transmute_ref!(self);

        slice
    }
}

impl PipelineSpec for MatmulSharedPipelineSpec {
    type PushConst = MatmulSharedPushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler = compiler::ShaderCompiler::new(MATMUL_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler.compile().unwrap()
    }

    fn get_spec_info(&self) -> (Box<[SpecializationMapEntry]>, Vec<u8>) {
        //offset needs to be the offset in the spec_buffer vec not the struct
        let spec_entrys = [
            SpecializationMapEntry {
                constant_id: 0,
                offset: 0,
                size: std::mem::size_of_val(&self.local_x),
            },
            SpecializationMapEntry {
                constant_id: 1,
                offset: 4,
                size: std::mem::size_of_val(&self.local_y),
            },
            SpecializationMapEntry {
                constant_id: 2,
                offset: 8,
                size: std::mem::size_of_val(&self.n),
            },
        ];

        let mut spec_buffer: Vec<u8> = Vec::new();
        let local_x = self.local_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&local_x);
        let local_y = self.local_y.to_ne_bytes();
        spec_buffer.extend_from_slice(&local_y);
        let n = self.n.to_ne_bytes();
        spec_buffer.extend_from_slice(&n);

        debug_assert!(spec_buffer.len() <= spec_entrys.iter().fold(0, |acc, x| acc + x.size));

        (Box::new(spec_entrys), spec_buffer)
    }

    fn build_pipeline(&self, inst: &VultenInstance) -> Arc<VultenPipeline> {
        let desc_types: Vec<vk::DescriptorType> = vec![vk::DescriptorType::STORAGE_BUFFER; 3];
        let shader = self.get_shader();
        let spec_info = self.get_spec_info();

        let pipe = inst
            .create_compute_pipeline(
                desc_types,
                &shader,
                Some(
                    &SpecializationInfo::default()
                        .map_entries(&spec_info.0)
                        .data(&spec_info.1),
                ),
                Self::PushConst::get_ranges(),
            )
            .unwrap();

        Arc::new(pipe)
    }
}

pub struct MatMulKernelShared<'a> {
    matmul: MatMulKernel<'a>,
    spec: Option<MatmulSharedPipelineSpec>,
}

impl<'a> MatMulKernelShared<'a> {
    pub fn new(matmul: MatMulKernel<'a>) -> Self {
        Self {
            matmul,
            spec: Default::default(),
        }
    }
}

impl<'a> MatMulKernelVersion<'a> for MatMulKernelShared<'a> {
    fn get_pipeline(&mut self) -> Result<Arc<VultenPipeline>, &'static str> {
        if let Some(spec) = self.spec.as_ref() {
            Ok(self
                .matmul
                .inst
                .get_pipeline_from_spec(PipelineSpecs::MatmulShared(spec.clone())))
        } else {
            let a_dims = self.matmul.a_dims.as_ref().ok_or("Missing a dims")?;
            let b_dims = self.matmul.b_dims.as_ref().ok_or("Missing b dims")?;

            let spec = MatmulSharedPipelineSpec {
                //local_x: self.matmul.inst.device_props.sub_group_size.max(1),
                local_x: 256,
                local_y: 1,
                n: a_dims[1] as u32,
                d_type: self.matmul.d_type,
            };
            let pipeline = self
                .matmul
                .inst
                .get_pipeline_from_spec(PipelineSpecs::MatmulShared(spec.clone()));
            self.spec = Some(spec);

            Ok(pipeline)
        }
    }

    fn get_descriptors(
        &mut self,
        pipeline: Arc<VultenPipeline>,
    ) -> Result<Vec<VultenDescriptor<'a>>, &'static str> {
        let descriptors = self
            .matmul
            .inst
            .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline)
            .or(Err("Could not get descriptor set"))?;

        let a_desc_buff = self
            .matmul
            .a
            .as_ref()
            .ok_or("No a operand")?
            .get_descriptor_info()?;
        let b_desc_buff = self
            .matmul
            .b
            .as_ref()
            .ok_or("No b operand")?
            .get_descriptor_info()?;
        let output_desc_buff = self
            .matmul
            .output
            .as_ref()
            .ok_or("No output operand")?
            .get_descriptor_info()?;

        let write_sets = [
            WriteDescriptorSet::default()
                .dst_set(descriptors.descriptor[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&a_desc_buff),
            WriteDescriptorSet::default()
                .dst_set(descriptors.descriptor[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&b_desc_buff),
            WriteDescriptorSet::default()
                .dst_set(descriptors.descriptor[0])
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&output_desc_buff),
        ];
        self.matmul.inst.update_descriptor_sets(&write_sets, &[]);

        Ok(vec![descriptors])
    }

    fn record<'b>(
        &mut self,
        mut builder: CommandBufferBuilder<'b>,
        pipeline: Arc<VultenPipeline>,
        descriptors: &[VultenDescriptor],
    ) -> Result<CommandBufferBuilder<'b>, &'static str> {
        let mut push = MatmulSharedPushConst::default();

        builder = builder
            .bind_pipeline(PipelineBindPoint::COMPUTE, pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                &descriptors[0].descriptor,
                &[],
            );

        let spec = self.spec.as_ref().ok_or("Missing spec")?;
        let a_dims = self.matmul.a_dims.as_ref().ok_or("Missing a dims")?;
        
                let threads_x = a_dims[1] as u32 / 128;
                let threads_y = a_dims[1] as u32 / 128;
                builder = builder
                    .push_constants(
                        pipeline.pipeline_layout,
                        ShaderStageFlags::COMPUTE,
                        0,
                        push.get_slice(),
                    )
                    .dispatch(threads_x, threads_y, 1);

        Ok(builder)
    }

    fn run(&mut self) -> Result<(), &'static str> {
        let pipeline = self.get_pipeline()?;
        let descriptors = self.get_descriptors(pipeline.clone())?;
        let q = self.matmul.inst.get_queue(QueueFlags::COMPUTE);
        let cmd_buffs = self
            .matmul
            .inst
            .create_cmd_buffers(1, &q)
            .or(Err("Could not create command buffers"))?;
        let builder = CommandBufferBuilder::new(cmd_buffs[0], &self.matmul.inst.device).begin();

        self.record(builder, pipeline, &descriptors)?
            .end()
            .build()?;

        let sub_info = SubmitInfo::default().command_buffers(&cmd_buffs);
        let fence = self
            .matmul
            .inst
            .create_fence()
            .or(Err("Could not create fence"))?;

        self.matmul
            .inst
            .submit_queue(&q, &[sub_info], fence)
            .or(Err("Could not submit queue"))?;
        self.matmul
            .inst
            .wait_for_fences(&[fence], true)
            .or(Err("Fence timed out"))?;

        self.matmul.inst.destroy_fence(fence);
        self.matmul.inst.free_cmd_buffers(&q, cmd_buffs);

        Ok(())
    }
}