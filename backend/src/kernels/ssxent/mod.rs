use ash::vk::{
    self, AccessFlags, DependencyFlags, DescriptorType, MemoryBarrier, PipelineBindPoint,
    PipelineStageFlags, PushConstantRange, QueueFlags, ShaderStageFlags, SpecializationInfo,
    SpecializationMapEntry, SubmitInfo, WriteDescriptorSet,
};
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    cmd_buff::CommandBufferBuilder,
    compiler,
    descriptor::VultenDescriptor,
    kernels::Chunkable,
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

use super::KernelBuff;

const SSXENT_SOURCE: &str = include_str!("ssxent.comp");
const OP_LOSS: u32 = 0;
const OP_GRAD: u32 = 1;

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct SsxentPipelineSpec {
    local_x: u32,
    d_type: VultenDataType,
    label_d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct SsxentPushConst {
    pub start: u32,
    pub stop: u32,
    pub num_logits: u32,
    pub op: u32,
}

impl PushConstSpec for SsxentPushConst {
    fn get_ranges() -> &'static [PushConstantRange] {
        &[PushConstantRange {
            offset: 0,
            stage_flags: ShaderStageFlags::COMPUTE,
            size: std::mem::size_of::<Self>() as u32,
        }]
    }

    #[inline]
    fn get_slice(&self) -> &[u8] {
        let slice: &[u8; 16] = zerocopy::transmute_ref!(self);

        slice
    }
}

impl PipelineSpec for SsxentPipelineSpec {
    type PushConst = SsxentPushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler = compiler::ShaderCompiler::new(SSXENT_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();
        compiler.add_type_spec(1, self.label_d_type).unwrap();

        compiler.compile().unwrap()
    }

    fn get_spec_info(&self) -> (Box<[SpecializationMapEntry]>, Vec<u8>) {
        //offset needs to be the offset in the spec_buffer vec not the struct
        let spec_entrys = [SpecializationMapEntry {
            constant_id: 0,
            offset: 0,
            size: std::mem::size_of_val(&self.local_x),
        }];

        let mut spec_buffer: Vec<u8> = Vec::new();
        let local_x_slice = self.local_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&local_x_slice);

        debug_assert!(spec_buffer.len() <= spec_entrys.iter().fold(0, |acc, x| acc + x.size));

        (Box::new(spec_entrys), spec_buffer)
    }

    fn build_pipeline(&self, inst: &VultenInstance) -> Arc<VultenPipeline> {
        let desc_types: Vec<vk::DescriptorType> = vec![vk::DescriptorType::STORAGE_BUFFER; 4];
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

pub struct SSXENTKernel<'a> {
    inst: &'a VultenInstance,
    d_type: VultenDataType,
    label_d_type: VultenDataType,
    scratch: Option<KernelBuff<'a>>,
    backprop: Option<KernelBuff<'a>>,
    backprop_dims: Option<&'a [i64]>,
    labels: Option<KernelBuff<'a>>,
    loss_fat: Option<KernelBuff<'a>>,
    grad: Option<KernelBuff<'a>>,
    spec: Option<SsxentPipelineSpec>,
}

impl<'a> SSXENTKernel<'a> {
    pub fn new(
        inst: &'a VultenInstance,
        d_type: VultenDataType,
        label_d_type: VultenDataType,
    ) -> Self {
        SSXENTKernel {
            inst,
            d_type,
            label_d_type,
            scratch: Default::default(),
            backprop: Default::default(),
            backprop_dims: Default::default(),
            labels: Default::default(),
            loss_fat: Default::default(),
            grad: Default::default(),
            spec: Default::default(),
        }
    }

    pub fn scratch(mut self, buff: KernelBuff<'a>) -> Result<Self, &'static str> {
        self.scratch = Some(buff);

        Ok(self)
    }

    pub fn backprop(mut self, buff: KernelBuff<'a>, dims: &'a [i64]) -> Result<Self, &'static str> {
        if dims.contains(&0) {
            return Err("Input backprop has a zero dim!");
        }
        self.backprop = Some(buff);
        self.backprop_dims = Some(dims);

        Ok(self)
    }

    pub fn labels(mut self, buff: KernelBuff<'a>) -> Result<Self, &'static str> {
        self.labels = Some(buff);

        Ok(self)
    }

    pub fn loss_fat(mut self, buff: KernelBuff<'a>) -> Result<Self, &'static str> {
        self.loss_fat = Some(buff);

        Ok(self)
    }

    pub fn grad(mut self, buff: KernelBuff<'a>) -> Result<Self, &'static str> {
        self.grad = Some(buff);

        Ok(self)
    }

    pub fn get_pipeline(&mut self) -> Result<Arc<VultenPipeline>, &'static str> {
        if let Some(spec) = self.spec.as_ref() {
            Ok(self
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Ssxent(spec.clone())))
        } else {
            let spec = SsxentPipelineSpec {
                local_x: self.inst.device_props.sub_group_size.max(1),
                d_type: self.d_type,
                label_d_type: self.label_d_type,
            };
            let pipeline = self
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Ssxent(spec.clone()));
            self.spec = Some(spec);

            Ok(pipeline)
        }
    }

    pub fn get_descriptors(
        &self,
        pipeline: Arc<VultenPipeline>,
    ) -> Result<Vec<VultenDescriptor<'a>>, &'static str> {
        let mut descriptors = Vec::new();

        let descriptors_loss = self
            .inst
            .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline.clone())
            .unwrap();
        let descriptors_grad = self
            .inst
            .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline.clone())
            .unwrap();

        let scratch_desc_buff = self
            .scratch
            .as_ref()
            .ok_or("No scratch operand")?
            .get_descriptor_info()?;
        let backprop_desc_buff = self
            .backprop
            .as_ref()
            .ok_or("No backprop operand")?
            .get_descriptor_info()?;
        let labels_desc_buff = self
            .labels
            .as_ref()
            .ok_or("No labels operand")?
            .get_descriptor_info()?;
        let loss_fat_desc_buff = self
            .loss_fat
            .as_ref()
            .ok_or("No loss_fat operand")?
            .get_descriptor_info()?;
        let grad_desc_buff = self
            .grad
            .as_ref()
            .ok_or("No grad operand")?
            .get_descriptor_info()?;

        let write_sets = [
            //loss
            WriteDescriptorSet::default()
                .dst_set(descriptors_loss.descriptor[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&scratch_desc_buff),
            WriteDescriptorSet::default()
                .dst_set(descriptors_loss.descriptor[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&backprop_desc_buff),
            WriteDescriptorSet::default()
                .dst_set(descriptors_loss.descriptor[0])
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&labels_desc_buff),
            WriteDescriptorSet::default()
                .dst_set(descriptors_loss.descriptor[0])
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&loss_fat_desc_buff),
            //Grad
            WriteDescriptorSet::default()
                .dst_set(descriptors_grad.descriptor[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&scratch_desc_buff),
            WriteDescriptorSet::default()
                .dst_set(descriptors_grad.descriptor[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&backprop_desc_buff),
            WriteDescriptorSet::default()
                .dst_set(descriptors_grad.descriptor[0])
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&labels_desc_buff),
            WriteDescriptorSet::default()
                .dst_set(descriptors_grad.descriptor[0])
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&grad_desc_buff),
        ];
        self.inst.update_descriptor_sets(&write_sets, &[]);

        descriptors.push(descriptors_loss);
        descriptors.push(descriptors_grad);

        Ok(descriptors)
    }

    pub fn record<'b>(
        &self,
        mut builder: CommandBufferBuilder<'b>,
        pipeline: Arc<VultenPipeline>,
        descriptors: &[VultenDescriptor],
    ) -> Result<CommandBufferBuilder<'b>, &'static str> {
        let mut push = SsxentPushConst::default();

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
        let chunk_size = self.inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
        let backprop_dims = self
            .backprop_dims
            .as_ref()
            .ok_or("Missing backprop_dims dims")?;
        let total_elements: i64 = backprop_dims.iter().product();
        push.num_logits = backprop_dims[1] as u32;
        let barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
            .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

        let chunks = (0..total_elements).as_chunks(chunk_size);

        //Loss
        push.op = OP_LOSS;
        for chunk in &chunks {
            push.start = chunk.start as u32;
            push.stop = chunk.end as u32;

            let threads = ((chunk.end - chunk.start) as f32 / spec.local_x as f32).ceil() as u32;
            builder = builder
                .push_constants(
                    pipeline.pipeline_layout,
                    ShaderStageFlags::COMPUTE,
                    0,
                    push.get_slice(),
                )
                .dispatch(threads, 1, 1);
        }

        builder = builder.pipeline_barrier(
            PipelineStageFlags::COMPUTE_SHADER,
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );

        //Grad
        push.op = OP_GRAD;
        builder = builder.bind_descriptor_sets(
            PipelineBindPoint::COMPUTE,
            pipeline.pipeline_layout,
            0,
            &descriptors[1].descriptor,
            &[],
        );
        for chunk in chunks {
            push.start = chunk.start as u32;
            push.stop = chunk.end as u32;

            let threads = ((chunk.end - chunk.start) as f32 / spec.local_x as f32).ceil() as u32;
            builder = builder
                .push_constants(
                    pipeline.pipeline_layout,
                    ShaderStageFlags::COMPUTE,
                    0,
                    push.get_slice(),
                )
                .dispatch(threads, 1, 1);
        }

        Ok(builder)
    }

    pub fn run(&mut self) -> Result<(), &'static str> {
        let pipeline = self.get_pipeline()?;
        let descriptors = self.get_descriptors(pipeline.clone())?;
        let q = self.inst.get_queue(QueueFlags::COMPUTE);
        let cmd_buffs = self
            .inst
            .create_cmd_buffers(1, &q)
            .or(Err("Could not create command buffers"))?;
        let builder = CommandBufferBuilder::new(cmd_buffs[0], &self.inst.device).begin();

        self.record(builder, pipeline, &descriptors)?
            .end()
            .build()?;

        let sub_info = SubmitInfo::default().command_buffers(&cmd_buffs);
        let fence = self.inst.create_fence().or(Err("Could not create fence"))?;

        self.inst
            .submit_queue(&q, &[sub_info], fence)
            .or(Err("Could not submit queue"))?;
        self.inst
            .wait_for_fences(&[fence], true)
            .or(Err("Fence timed out"))?;

        self.inst.destroy_fence(fence);
        self.inst.free_cmd_buffers(&q, cmd_buffs);

        Ok(())
    }
}
