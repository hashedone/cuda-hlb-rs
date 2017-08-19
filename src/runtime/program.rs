use super::ffi;
use super::Result;
use super::context::Context;
use std;
use std::mem::uninitialized;

pub struct ProgramBuilder<'a> {
    context: &'a Context
}

impl<'a> ProgramBuilder<'a> {
    pub(super) fn new(context: &'a Context) -> ProgramBuilder <'a> {
        ProgramBuilder { context }
    }
}

pub struct Program<'a> {
    handle: ffi::CUmodule,
    context: &'a Context
}

impl<'a> ProgramBuilder<'a> {
    #[cfg(feature="compiler")]
    pub fn from_compiled(self, program: ::compiler::Program) -> Result<Program<'a>> {
        let data = program.ptx().map_err(|_| super::result::Error::Unknown)?;
        unsafe {
            let mut handle = uninitialized();
            self.context.make_curent()?;
            ffi::cuModuleLoadData(&mut handle, data.as_ptr() as *const _)?;
            Ok(Program { handle, context: self.context })
        }
    }

    pub fn from_file<S: Into<String>>(self, fname: S) -> Result<Program<'a>> {
        unsafe {
            let fname = std::ffi::CString::new(fname.into()).unwrap();
            let mut handle = uninitialized();
            self.context.make_curent()?;
            ffi::cuModuleLoad(&mut handle, fname.as_ptr())?;
            Ok(Program { handle, context: self.context })
        }
    }

    pub fn from_data<S: Into<String>>(self, data: S) -> Result<Program<'a>> {
        unsafe {
            let data = std::ffi::CString::new(data.into()).unwrap();
            let mut handle = uninitialized();
            self.context.make_curent()?;
            ffi::cuModuleLoadData(&mut handle, data.as_ptr() as *const _)?;
            Ok(Program { handle, context: self.context })
        }
    }
}

impl<'a> Program<'a> {
}
