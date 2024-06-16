use shaderc::{
    CompilationArtifact, CompileOptions, Compiler, IncludeCallbackResult, ResolvedInclude,
};

use crate::{VultenDataType, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, VK_ENV_VER};

const SHADER_PRELUDE: &str = include_str!("prelude.h");
const BINARY: &str = include_str!("kernels/binary/binary.h");

const FLOAT_NUM: &str = "0";
const INT_NUM: &str = "1";
const UINT_NUM: &str = "2";
const INT64_NUM: &str = "3";
const UINT64_NUM: &str = "4";

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
            "binary.h" => IncludeCallbackResult::Ok(ResolvedInclude {
                resolved_name: file.to_string(),
                content: BINARY.to_string(),
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
                self.opts
                    .add_macro_definition(&format!("TYPE_NUM_{:}", num), Some(FLOAT_NUM));
                Ok(())
            }
            DT_INT32 => {
                self.opts
                    .add_macro_definition(&format!("TYPE_{:}", num), Some("int"));
                self.opts
                    .add_macro_definition(&format!("TYPE_P_{:}", num), Some("highp int"));
                self.opts
                    .add_macro_definition(&format!("TYPE_NUM_{:}", num), Some(INT_NUM));
                Ok(())
            }
            DT_UINT32 => {
                self.opts
                    .add_macro_definition(&format!("TYPE_{:}", num), Some("uint"));
                self.opts
                    .add_macro_definition(&format!("TYPE_P_{:}", num), Some("highp uint"));
                self.opts
                    .add_macro_definition(&format!("TYPE_NUM_{:}", num), Some(UINT_NUM));
                Ok(())
            }
            DT_INT64 => {
                self.opts
                    .add_macro_definition(&format!("TYPE_{:}", num), Some("int64_t"));
                self.opts
                    .add_macro_definition(&format!("TYPE_P_{:}", num), Some("int64_t"));
                self.opts.add_macro_definition("USE_INT64", None);
                self.opts
                    .add_macro_definition(&format!("TYPE_NUM_{:}", num), Some(INT64_NUM));
                Ok(())
            }
            DT_UINT64 => {
                self.opts
                    .add_macro_definition(&format!("TYPE_{:}", num), Some("uint64_t"));
                self.opts
                    .add_macro_definition(&format!("TYPE_P_{:}", num), Some("int64_t"));
                self.opts.add_macro_definition("USE_INT64", None);
                self.opts
                    .add_macro_definition(&format!("TYPE_NUM_{:}", num), Some(UINT64_NUM));
                Ok(())
            }
            _ => Err("Invalid type"),
        }
    }
}
