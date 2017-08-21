use super::Result;
use super::buffer::BufferBuilder;
use super::ffi;
use super::program::ProgramBuilder;
use std::ops::Deref;

#[derive(Copy, Clone)]
pub struct Context {
    handle: ffi::CUcontext,
}

impl Context {
    pub(super) fn make_current(&self) -> Result<()> {
        unsafe {
            use std::ops::Try;
            ffi::cuCtxSetCurrent(self.handle).into_result()
        }
    }

    pub fn program_builder(&self) -> ProgramBuilder {
        ProgramBuilder::new(self)
    }

    pub fn buffer_builder<T: Sized + Copy>(&self) -> BufferBuilder<T> {
        BufferBuilder::new(self)
    }

    // TODO: wrap state management functions
    // TODO: wrap context reseting
}

pub struct PrimaryContext {
    device: ffi::CUdevice,
    ctx: Context,
}

impl PrimaryContext {
    pub(super) fn new(device: ffi::CUdevice, ctx: ffi::CUcontext) -> PrimaryContext {
        PrimaryContext {
            device,
            ctx: Context { handle: ctx },
        }
    }
}

impl Drop for PrimaryContext {
    fn drop(&mut self) {
        unsafe {
            ffi::cuDevicePrimaryCtxRelease(self.device);
        }
    }
}

impl Deref for PrimaryContext {
    type Target = Context;

    fn deref(&self) -> &Context {
        &self.ctx
    }
}

// TODO: wrap Context Management functions
