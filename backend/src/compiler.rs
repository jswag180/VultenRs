use shaderc::{
    CompilationArtifact, CompileOptions, Compiler, IncludeCallbackResult, ResolvedInclude,
};

use crate::{VultenDataType, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, VK_ENV_VER};

const SHADER_PRELUDE: &str = include_str!("prelude.h");

pub struct ShaderCompiler<'a> {
    name: &'static str,
    compiler: Compiler,
    pub opts: CompileOptions<'a>,
    source: &'static str,
}

impl ShaderCompiler<'_> {
    pub fn new(name: &'static str, source: &'static str) -> Self {
        let mut opts = CompileOptions::new().expect("Filed to init compiler opts");
        opts.set_optimization_level(shaderc::OptimizationLevel::Performance);
        opts.set_source_language(shaderc::SourceLanguage::GLSL);
        opts.set_target_env(shaderc::TargetEnv::Vulkan, VK_ENV_VER);
        opts.set_include_callback(|file, _inc_type, from, _depth| match file {
            "prelude.h" => IncludeCallbackResult::Ok(ResolvedInclude {
                resolved_name: file.to_string(),
                content: SHADER_PRELUDE.to_string(),
            }),
            _ => IncludeCallbackResult::Err(format!(
                "Include not found for {:} from {:}",
                file, from
            )),
        });

        Self {
            name,
            source,
            compiler: Compiler::new().expect("Filed to init compiler"),
            opts,
        }
    }

    pub fn compile(self) -> CompilationArtifact {
        self.compiler
            .compile_into_spirv(
                self.source,
                shaderc::ShaderKind::Compute,
                self.name,
                "main",
                Some(&self.opts),
            )
            .unwrap()
    }

    pub fn add_type_spec(&mut self, num: i32, d_type: VultenDataType) -> Result<(), &'static str> {
        match d_type {
            DT_FLOAT => {
                self.opts
                    .add_macro_definition(&format!("TYPE_{:}", num), Some("float"));
                self.opts
                    .add_macro_definition(&format!("TYPE_P_{:}", num), Some("highp float"));
                Ok(())
            }
            DT_INT32 => {
                self.opts
                    .add_macro_definition(&format!("TYPE_{:}", num), Some("int"));
                self.opts
                    .add_macro_definition(&format!("TYPE_P_{:}", num), Some("highp int"));
                Ok(())
            }
            DT_UINT32 => {
                self.opts
                    .add_macro_definition(&format!("TYPE_{:}", num), Some("uint"));
                self.opts
                    .add_macro_definition(&format!("TYPE_P_{:}", num), Some("highp uint"));
                Ok(())
            }
            DT_INT64 => {
                self.opts
                    .add_macro_definition(&format!("TYPE_{:}", num), Some("int64_t"));
                self.opts
                    .add_macro_definition(&format!("TYPE_P_{:}", num), Some("int64_t"));
                self.opts.add_macro_definition("USE_INT64", None);
                Ok(())
            }
            DT_UINT64 => {
                self.opts
                    .add_macro_definition(&format!("TYPE_{:}", num), Some("uint64_t"));
                self.opts
                    .add_macro_definition(&format!("TYPE_P_{:}", num), Some("int64_t"));
                self.opts.add_macro_definition("USE_INT64", None);
                Ok(())
            }
            _ => Err("Invalid type"),
        }
    }
}
