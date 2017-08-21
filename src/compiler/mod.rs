pub mod result;

pub use self::result::Result;

use ffi::compiler as ffi;
use std;
use std::mem::uninitialized;

pub fn version() -> Result<(i32, i32)> {
    unsafe {
        let mut maj = uninitialized();
        let mut min = uninitialized();

        ffi::nvrtcVersion(&mut maj, &mut min)?;
        Ok((maj, min))
    }
}

#[derive(Clone, Copy, Debug)]
pub enum GpuArch {
    Compute20,
    Compute30,
    Compute35,
    Compute50,
    Compute52,
    Compute53,
}

pub struct Compiler {
    src: String,
    name: String,
    headers: Vec<(std::string::String, std::string::String)>,
    flags: Vec<std::string::String>,
    name_expressions: Vec<std::string::String>,
}

pub struct Program {
    prog: ffi::nvrtcProgram,
}

impl Compiler {
    pub fn new() -> Compiler {
        Compiler {
            src: "".into(),
            name: "".into(),
            headers: vec![],
            flags: vec![],
            name_expressions: vec![],
        }
    }

    pub fn src<S: Into<String>>(&mut self, src: S) -> &mut Compiler {
        self.src = src.into();
        self
    }

    pub fn name<S: Into<String>>(&mut self, name: S) -> &mut Compiler {
        self.name = name.into();
        self
    }

    pub fn header<SN: Into<String>, SS: Into<String>>(
        &mut self,
        name: SN,
        src: SS,
    ) -> &mut Compiler {
        self.headers.push((name.into(), src.into()));
        self
    }

    pub fn arch(&mut self, arch: GpuArch) -> &mut Compiler {
        let flag = match arch {
            GpuArch::Compute20 => "-arch compute_20",
            GpuArch::Compute30 => "-arch compute_30",
            GpuArch::Compute35 => "-arch compute_35",
            GpuArch::Compute50 => "-arch compute_50",
            GpuArch::Compute52 => "-arch compute_52",
            GpuArch::Compute53 => "-arch compute_53",
        };
        self.flags.push(flag.into());
        self
    }

    pub fn relocatable(&mut self, val: bool) -> &mut Compiler {
        let flag = if val { "-dc" } else { "-dw" };
        self.flags.push(flag.into());
        self
    }

    pub fn debug(&mut self) -> &mut Compiler {
        self.flags.push("-G".into());
        self.flags.push("-lineinfo".into());
        self
    }

    pub fn define(&mut self, name: &str, def: &str) -> &mut Compiler {
        let flag = format!("-D {}={}", name, def);
        self.flags.push(flag);
        self
    }

    pub fn include_path(&mut self, path: &str) -> &mut Compiler {
        let flag = format!("-I {}", path);
        self.flags.push(flag);
        self
    }

    pub fn preinclude(&mut self, header: &str) -> &mut Compiler {
        let flag = format!("-include {}", header);
        self.flags.push(flag);
        self
    }

    pub fn flag<S: Into<String>>(&mut self, flag: S) -> &mut Compiler {
        self.flags.push(flag.into());
        self
    }

    pub fn add_name_expression<S: Into<String>>(&mut self, expr: S) -> &mut Compiler {
        self.name_expressions.push(expr.into());
        self
    }

    pub fn compile(&self) -> Result<Program> {
        let src = std::ffi::CString::new(self.src.clone()).unwrap();
        let name = std::ffi::CString::new(self.name.clone()).unwrap();
        let headers_cnt = self.headers.len();
        let headers = self.headers
            .iter()
            .map(|&(ref name, ref src)| {
                (
                    std::ffi::CString::new(name.clone()).unwrap(),
                    std::ffi::CString::new(src.clone()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let (headers_names, headers_srcs): (Vec<_>, Vec<_>) = headers
            .iter()
            .map(|&(ref name, ref src)| (name.as_ptr(), src.as_ptr()))
            .unzip();
        let flags_cnt = self.flags.len();
        let flags = self.flags
            .iter()
            .map(|f| std::ffi::CString::new(f.clone()).unwrap())
            .collect::<Vec<_>>();
        let flags_p = flags.iter().map(|f| f.as_ptr()).collect::<Vec<_>>();

        unsafe {
            let mut prog = uninitialized();
            ffi::nvrtcCreateProgram(
                &mut prog,
                src.as_ptr(),
                name.as_ptr(),
                headers_cnt as i32,
                headers_srcs.as_ptr(),
                headers_names.as_ptr(),
            )?;

            for expr in self.name_expressions.iter().cloned() {
                let cexpr = std::ffi::CString::new(expr).unwrap();
                ffi::nvrtcAddNameExpression(prog, cexpr.as_ptr())?;
            }

            ffi::nvrtcCompileProgram(prog, flags_cnt as i32, flags_p.as_ptr())?;

            Ok(Program { prog })
        }
    }
}

impl Program {
    pub fn ptx(&self) -> Result<Vec<i8>> {
        unsafe {
            let mut ptx_size = uninitialized();
            ffi::nvrtcGetPTXSize(self.prog, &mut ptx_size)?;

            let mut ptx = vec![uninitialized(); ptx_size as usize];
            ffi::nvrtcGetPTX(self.prog, ptx.as_mut_ptr())?;

            Ok(ptx)
        }
    }

    pub fn log(&self) -> Result<String> {
        unsafe {
            let mut log_size = uninitialized();
            ffi::nvrtcGetProgramLogSize(self.prog, &mut log_size)?;

            let mut log = vec![uninitialized(); log_size as usize];
            ffi::nvrtcGetProgramLog(self.prog, log.as_mut_ptr())?;

            Ok(
                std::ffi::CStr::from_ptr(log.as_ptr())
                    .to_string_lossy()
                    .into(),
            )
        }
    }

    pub fn get_mangled(&self, name: &str) -> Result<String> {
        unsafe {
            let mut mangled = uninitialized();
            let name = std::ffi::CString::new(name).unwrap();
            ffi::nvrtcGetLoweredName(self.prog, name.as_ptr(), &mut mangled)?;
            Ok(std::ffi::CStr::from_ptr(mangled).to_string_lossy().into())
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn getting_version_works() {
        super::version().unwrap();
    }

    #[test]
    fn simple_program_compiles() {
        let program = r#"__global__ void hello(int *a, int *b)
            {
                a[threadIdx.x] += b[threadIdx.x];
            }"#;

        let prog = super::Compiler::new()
            .src(program)
            .add_name_expression("hello")
            .compile()
            .unwrap();

        assert!(prog.ptx().unwrap().len() > 0);
        prog.log().unwrap();
        assert!(prog.get_mangled("hello").unwrap().len() > 0);
    }
}
