use ash::vk::{
    self, DescriptorType, PipelineBindPoint, PushConstantRange, QueueFlags, ShaderStageFlags,
    SpecializationInfo, SpecializationMapEntry, SubmitInfo, WriteDescriptorSet,
};
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    cmd_buff::CommandBufferBuilder,
    compiler,
    kernels::Chunkable,
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    utills::calculate_strdies,
    VultenDataType, VultenInstance,
};

use super::{BinaryKernel, BinaryKernelVersion};

const BINARY_BROAD_SOURCE: &str = include_str!("binary_broad.comp");

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct BinaryBroadPipelineSpec {
    local_x: u32,
    op: super::BinaryOp,
    x_dims: [i64; 9],
    y_dims: [i64; 9],
    strides: [i64; 9],
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct BinarayBroadPushConst {
    start: u32,
    stop: u32,
}

impl PushConstSpec for BinarayBroadPushConst {
    fn get_ranges() -> &'static [PushConstantRange] {
        &[PushConstantRange {
            offset: 0,
            stage_flags: ShaderStageFlags::COMPUTE,
            size: std::mem::size_of::<Self>() as u32,
        }]
    }

    #[inline]
    fn get_slice(&self) -> &[u8] {
        let slice: &[u8; 8] = zerocopy::transmute_ref!(self);

        slice
    }
}

impl PipelineSpec for BinaryBroadPipelineSpec {
    type PushConst = BinarayBroadPushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new(BINARY_BROAD_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        for (i, (dim_x, dim_y)) in self.x_dims.iter().zip(self.y_dims.iter()).enumerate() {
            compiler.add_define(format!("DIM_X_{i:}"), Some(dim_x.to_string()));
            compiler.add_define(format!("DIM_Y_{i:}"), Some(dim_y.to_string()));
        }

        let strides_str = self.strides.iter().fold("".to_string(), |mut acc, x| {
            acc += &(x.to_string() + ",");
            acc
        });
        compiler.add_define(
            "STRIDES_ARR".into(),
            Some(strides_str.strip_suffix(',').unwrap().into()),
        );

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
                size: std::mem::size_of::<u32>(),
            },
        ];

        let mut spec_buffer: Vec<u8> = Vec::new();
        let local_x_slice = self.local_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&local_x_slice);
        let op_as_u32: u32 = self.op as u32;
        let op_slice = op_as_u32.to_ne_bytes();
        spec_buffer.extend_from_slice(&op_slice);

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

pub struct BinaryKernelBroad<'a> {
    binary: BinaryKernel<'a>,
    spec: Option<BinaryBroadPipelineSpec>,
}

impl<'a> BinaryKernelBroad<'a> {
    pub fn new(binary: BinaryKernel<'a>) -> Self {
        Self {
            binary,
            spec: Default::default(),
        }
    }
}

impl<'a> BinaryKernelVersion<'a> for BinaryKernelBroad<'a> {
    fn get_pipeline(&mut self) -> Result<Arc<VultenPipeline>, &'static str> {
        if let Some(spec) = self.spec.as_ref() {
            Ok(self
                .binary
                .inst
                .get_pipeline_from_spec(PipelineSpecs::BinaryBroad(Box::new(spec.clone()))))
        } else {
            let mut x_dims: [i64; 9] = [1; 9];
            x_dims[9 - self.binary.a_dims.len()..].clone_from_slice(&self.binary.a_dims);

            let mut y_dims: [i64; 9] = [1; 9];
            y_dims[9 - self.binary.b_dims.len()..].clone_from_slice(&self.binary.b_dims);

            let mut out_dims: [i64; 9] = [1; 9];
            out_dims[9 - self.binary.output_dims.len()..]
                .clone_from_slice(&self.binary.output_dims);

            let mut strides_padded: [i64; 9] = [1; 9];
            let strides = calculate_strdies(&out_dims);
            strides_padded.clone_from_slice(&strides[1..]);
            let spec = BinaryBroadPipelineSpec {
                local_x: self.binary.inst.device_props.sub_group_size.max(1),
                op: self.binary.op,
                x_dims,
                y_dims,
                strides: strides_padded,
                d_type: self.binary.d_type,
            };
            let pipeline = self
                .binary
                .inst
                .get_pipeline_from_spec(PipelineSpecs::BinaryBroad(Box::new(spec.clone())));
            self.spec = Some(spec);

            Ok(pipeline)
        }
    }

    fn get_descriptors(
        &mut self,
        pipeline: Arc<VultenPipeline>,
    ) -> Result<Vec<crate::descriptor::VultenDescriptor<'a>>, &'static str> {
        let descriptors = self
            .binary
            .inst
            .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline)
            .or(Err("Could not get descriptor set"))?;

        let a_desc_buff = self
            .binary
            .a
            .as_ref()
            .ok_or("No a operand")?
            .get_descriptor_info()?;
        let b_desc_buff = self
            .binary
            .b
            .as_ref()
            .ok_or("No b operand")?
            .get_descriptor_info()?;
        let output_desc_buff = self
            .binary
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
        self.binary.inst.update_descriptor_sets(&write_sets, &[]);

        Ok(vec![descriptors])
    }

    fn record<'b>(
        &mut self,
        mut builder: CommandBufferBuilder<'b>,
        pipeline: Arc<VultenPipeline>,
        descriptors: &[crate::descriptor::VultenDescriptor],
    ) -> Result<CommandBufferBuilder<'b>, &'static str> {
        let mut push = BinarayBroadPushConst::default();

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
        let chunk_size =
            self.binary.inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
        let total_elements: i64 = self.binary.output_dims.iter().product();
        let chunks = (0..total_elements).as_chunks(chunk_size).into_iter();

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

    fn run(&mut self) -> Result<(), &'static str> {
        let pipeline = self.get_pipeline()?;
        let descriptors = self.get_descriptors(pipeline.clone())?;
        let q = self.binary.inst.get_queue(QueueFlags::COMPUTE);
        let cmd_buffs = self
            .binary
            .inst
            .create_cmd_buffers(1, &q)
            .or(Err("Could not create command buffers"))?;
        let builder = CommandBufferBuilder::new(cmd_buffs[0], &self.binary.inst.device).begin();

        self.record(builder, pipeline, &descriptors)?
            .end()
            .build()?;

        let sub_info = SubmitInfo::default().command_buffers(&cmd_buffs);
        let fence = self
            .binary
            .inst
            .create_fence()
            .or(Err("Could not create fence"))?;

        self.binary
            .inst
            .submit_queue(&q, &[sub_info], fence)
            .or(Err("Could not submit queue"))?;
        self.binary
            .inst
            .wait_for_fences(&[fence], true)
            .or(Err("Fence timed out"))?;

        self.binary.inst.destroy_fence(fence);
        self.binary.inst.free_cmd_buffers(&q, cmd_buffs);

        Ok(())
    }
}
