use super::Program;
use super::Result;
use super::ffi;
use std;
use std::mem::uninitialized;

pub struct Kernel<'a> {
    handle: ffi::CUfunction,
    program: &'a Program<'a>,
}

impl<'a> Kernel<'a> {
    pub(super) fn new(program: &'a Program, name: &std::ffi::CString) -> Result<Kernel<'a>> {
        unsafe {
            let mut handle = uninitialized();
            program.context.make_current()?;
            ffi::cuModuleGetFunction(&mut handle, program.handle, name.as_ptr())?;
            Ok(Kernel { handle, program })
        }
    }
}
