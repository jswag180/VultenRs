use glslang::{
    error::GlslangError, Compiler, CompilerOptions, Program, ShaderInput, ShaderSource,
    ShaderStage, SourceLanguage, Target,
};

use crate::{
    VultenDataType, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, VK_ENV_VER, VK_SPIRV_VER,
};

const SHADER_PRELUDE: &str = include_str!("prelude.h");
const BINARY: &str = include_str!("kernels/binary/binary.h");

const FLOAT_NUM: &str = "0";
const INT_NUM: &str = "1";
const UINT_NUM: &str = "2";
const INT64_NUM: &str = "3";
const UINT64_NUM: &str = "4";

pub struct ShaderCompiler {
    opts: CompilerOptions,
    defs: Vec<(String, Option<String>)>,
    source: &'static str,
}

impl ShaderCompiler {
    pub fn new(source: &'static str) -> Self {
        let opts = CompilerOptions {
            target: Target::Vulkan {
                version: VK_ENV_VER,
                spirv_version: VK_SPIRV_VER,
            },

            source_language: SourceLanguage::GLSL,
            ..Default::default()
        };

        Self {
            source,
            defs: Vec::new(),
            opts,
        }
    }

    pub fn compile(self) -> Result<Vec<u32>, GlslangError> {
        let compiler = Compiler::acquire().unwrap();
        let source = ShaderSource::from(self.source);

        let mut inc = IncludeHandler {};

        let defines: Vec<(&str, Option<&str>)> = self
            .defs
            .iter()
            .map(|def| (def.0.as_str(), def.1.as_deref()))
            .collect();

        let input = ShaderInput::new(
            &source,
            ShaderStage::Compute,
            &self.opts,
            Some(&defines),
            Some(&mut inc),
        )?;
        let shader = glslang::Shader::new(compiler, input)?;

        let mut program = Program::new(compiler);
        program.add_shader(&shader);

        program.compile(ShaderStage::Compute)
    }

    pub fn add_define(&mut self, key: String, val: Option<String>) {
        self.defs.push((key, val));
    }

    pub fn add_type_spec(&mut self, num: i32, d_type: VultenDataType) -> Result<(), &'static str> {
        match d_type {
            DT_FLOAT => {
                self.add_define(format!("TYPE_{num:}"), Some("float".to_string()));
                self.add_define(format!("TYPE_P_{num:}"), Some("highp float".into()));
                self.add_define(format!("TYPE_NUM_{num:}"), Some(FLOAT_NUM.into()));
                self.add_define(format!("TYPE_MAX_{num:}"), Some("1.0 / 0.0".to_string()));
                self.add_define(
                    format!("TYPE_MIN_{num:}"),
                    Some("-(1.0 / 0.0)".to_string()),
                );
                Ok(())
            }
            DT_INT32 => {
                self.add_define(format!("TYPE_{num:}"), Some("int".into()));
                self.add_define(format!("TYPE_P_{num:}"), Some("highp int".into()));
                self.add_define(format!("TYPE_NUM_{num:}"), Some(INT_NUM.into()));
                self.add_define(
                    format!("TYPE_MAX_{num:}"),
                    Some("~0 ^ 1 << 31".to_string()),
                );
                self.add_define(format!("TYPE_MIN_{num:}"), Some("1 << 31".to_string()));
                Ok(())
            }
            DT_UINT32 => {
                self.add_define(format!("TYPE_{num:}"), Some("uint".into()));
                self.add_define(format!("TYPE_P_{num:}"), Some("highp uint".into()));
                self.add_define(format!("TYPE_NUM_{num:}"), Some(UINT_NUM.into()));
                self.add_define(format!("TYPE_MAX_{num:}"), Some("~0".to_string()));
                self.add_define(format!("TYPE_MIN_{num:}"), Some("0".to_string()));
                Ok(())
            }
            DT_INT64 => {
                self.add_define(format!("TYPE_{num:}"), Some("int64_t".into()));
                self.add_define(format!("TYPE_P_{num:}"), Some("int64_t".into()));
                self.add_define("USE_INT64".into(), None);
                self.add_define(format!("TYPE_NUM_{num:}"), Some(INT64_NUM.into()));
                self.add_define(
                    format!("TYPE_MAX_{num:}"),
                    Some("~0 ^ 1 << 63".to_string()),
                );
                self.add_define(format!("TYPE_MIN_{num:}"), Some("1 << 63".to_string()));
                Ok(())
            }
            DT_UINT64 => {
                self.add_define(format!("TYPE_{num:}"), Some("uint64_t".into()));
                self.add_define(format!("TYPE_P_{num:}"), Some("int64_t".into()));
                self.add_define("USE_INT64".into(), None);
                self.add_define(format!("TYPE_NUM_{num:}"), Some(UINT64_NUM.into()));
                self.add_define(format!("TYPE_MAX_{num:}"), Some("~0".to_string()));
                self.add_define(format!("TYPE_MIN_{num:}"), Some("0".to_string()));
                Ok(())
            }
            _ => Err("Invalid type"),
        }
    }
}

struct IncludeHandler;
impl glslang::include::IncludeHandler for IncludeHandler {
    fn include(
        &mut self,
        _ty: glslang::include::IncludeType,
        header_name: &str,
        includer_name: &str,
        _include_depth: usize,
    ) -> Option<glslang::include::IncludeResult> {
        match header_name {
            "prelude.h" => Some(glslang::include::IncludeResult {
                name: header_name.to_string(),
                data: SHADER_PRELUDE.to_string(),
            }),
            "binary.h" => Some(glslang::include::IncludeResult {
                name: header_name.to_string(),
                data: BINARY.to_string(),
            }),
            _ => {
                println!(
                    "Failed to get include {includer_name:} for {header_name:}"
                );
                None
            }
        }
    }
}
