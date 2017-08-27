#![feature(try_trait)]
#![feature(conservative_impl_trait)]
#![feature(try_from)]
#![feature(collections_range)]
#![feature(trace_macros)]
#![feature(concat_idents)]
#![feature(plugin)]
#![recursion_limit = "1024"]

#![plugin(interpolate_idents)]

extern crate libc;
#[macro_use]
extern crate error_chain;

pub mod ffi;
pub mod result;
pub mod module;
pub mod buffer;

pub use self::buffer::Buffer;
pub use self::result::Result;

use std::mem::uninitialized;

static INIT: std::sync::Once = std::sync::ONCE_INIT;

pub fn init() {
    unsafe { INIT.call_once(|| { ffi::cuInit(0); }) };
}

enum Context {
    Primary(ffi::CUdevice, ffi::CUcontext),
}

pub struct Cuda {
    context: Context,
}

impl Cuda {
    pub fn with_primary_context() -> Result<Cuda> {
        init();
        unsafe {
            let mut devcnt = uninitialized();
            ffi::cuDeviceGetCount(&mut devcnt)?;
            if devcnt < 1 {
                bail!(result::ErrorKind::NoDevices)
            }
        }
        unsafe {
            let mut device = uninitialized();
            ffi::cuDeviceGet(&mut device, 0)?;
            let mut context = uninitialized();
            ffi::cuDevicePrimaryCtxRetain(&mut context, device)?;
            let context = Context::Primary(device, context);
            Ok(Cuda { context })
        }
    }

    pub fn make_current(&self) -> Result<ffi::CUcontext> {
        let context = match self.context {
            Context::Primary(_, context) => context,
        };

        unsafe {
            ffi::cuCtxSetCurrent(context)?;
        }
        Ok(context)
    }

    pub fn load_module<'a, Module: module::Module<'a>>(&'a self) -> Result<Module> {
        Module::load(self)
    }

    pub unsafe fn uninitialized_buffer<T: Sized>(
        &self,
        cnt: usize,
    ) -> Result<Buffer<T>> {
        Buffer::uninitialized(self, cnt)
    }

    pub fn new_buffer<T: Sized>(&self, cnt: usize) -> Result<Buffer<T>> {
        Buffer::new(self, cnt)
    }

    pub fn buffer_from_slice<'a, T: Sized + Copy>(&'a self, data: &[T]) -> Result<Buffer<'a, T>> {
        let buf = unsafe { Buffer::uninitialized(self, data.len())? };
        buf.load_slice(data)?;
        Ok(buf)
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
        match self.context {
            Context::Primary(device, _) => unsafe {
                ffi::cuDevicePrimaryCtxRelease(device);
            },
        }
    }
}
