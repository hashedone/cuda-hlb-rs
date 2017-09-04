#![feature(try_trait)]
#![feature(conservative_impl_trait)]
#![feature(try_from)]
#![feature(collections_range)]
#![feature(trace_macros)] #![feature(concat_idents)]
#![feature(plugin)]
#![recursion_limit = "1024"]

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

pub unsafe trait CudaPrim: Sized { }

unsafe impl CudaPrim for u8 { }
unsafe impl CudaPrim for u16 { }
unsafe impl CudaPrim for u32 { }
unsafe impl CudaPrim for u64 { }
unsafe impl CudaPrim for i8 { }
unsafe impl CudaPrim for i16 { }
unsafe impl CudaPrim for i32 { }
unsafe impl CudaPrim for i64 { }
unsafe impl CudaPrim for f32 { }
unsafe impl CudaPrim for f64 { }

pub unsafe trait AsCudaType<T> {
    type Type: Sized;

    unsafe fn cuda_type(&self) -> *const Self::Type;
}

unsafe impl<T: CudaPrim> AsCudaType<T> for T {
    type Type = T;
    unsafe fn cuda_type(&self) -> *const T { std::mem::transmute(self) }
}

#[derive(PartialEq, Debug)]
pub struct ExecProp {
    pub grid_dim: [usize; 3],
    pub block_dim: [usize; 3]
}

#[macro_export]
macro_rules! cuda_grid {
    ($x:expr, $y:expr, $z:expr) => { $crate::ExecProp { grid_dim: [$x, $y, $z], block_dim: [1, 1, 1] } };
    ($x:expr, $y:expr) => { cuda_grid!{$x, $y, 1} };
    ($x:expr) => { cuda_grid!{$x, 1, 1} };
}

#[macro_export]
macro_rules! cuda_block {
    ($x:expr, $y:expr, $z:expr) => { $crate::ExecProp { grid_dim: [1, 1, 1], block_dim: [$x, $y, $z] } };
    ($x:expr, $y:expr) => { cuda_block!{$x, $y, 1} };
    ($x:expr) => { cuda_block!{$x, 1, 1} };
}

#[macro_export]
macro_rules! cuda_ep_impl {
    ($ep:expr) => { $ep };
    ($ep:expr, grid($x:expr, $y:expr, $z:expr) $($tail:tt)*) => {
        cuda_ep_impl!{
            {
                let mut ep = $ep;
                ep.grid_dim = [$x, $y, $z];
                ep
            } $($tail)*
        }
    };
    ($ep:expr, grid($x:expr, $y:expr) $($tail:tt)*) => { cuda_ep_impl!{$ep, grid($x, $y, 1) $($tail)*} };
    ($ep:expr, grid($x:expr) $($tail:tt)*) => { cuda_ep_impl!{$ep, grid($x, 1, 1) $($tail)*} };
    ($ep:expr, block($x:expr, $y:expr, $z:expr) $($tail:tt)*) => {
        cuda_ep_impl!{
            {
                let mut ep = $ep;
                ep.block_dim = [$x, $y, $z];
                ep
            } $($tail)*
        }
    };
    ($ep:expr, block($x:expr, $y:expr) $($tail:tt)*) => { cuda_ep_impl!{$ep, block($x, $y, 1) $($tail)*} };
    ($ep:expr, block($x:expr) $($tail:tt)*) => { cuda_ep_impl!{$ep, block($x, 1, 1) $($tail)*} };
}

#[macro_export]
macro_rules! cuda_ep {
    ($($t:tt)+) => { cuda_ep_impl! { $crate::ExecProp { grid_dim: [1, 1, 1], block_dim: [1, 1, 1] }, $($t)* } }
}

#[cfg(test)]
mod test {
#[test]
fn cuda_grid_works() {
    assert_eq!(super::ExecProp { grid_dim: [2, 3, 4], block_dim: [1, 1, 1] }, cuda_grid!(2, 3, 4));
    assert_eq!(super::ExecProp { grid_dim: [2, 3, 1], block_dim: [1, 1, 1] }, cuda_grid!(2, 3));
    assert_eq!(super::ExecProp { grid_dim: [2, 1, 1], block_dim: [1, 1, 1] }, cuda_grid!(2));
}

#[test]
fn cuda_block_works() {
    assert_eq!(super::ExecProp { grid_dim: [1, 1, 1], block_dim: [2, 3, 4] }, cuda_block!(2, 3, 4));
    assert_eq!(super::ExecProp { grid_dim: [1, 1, 1], block_dim: [2, 3, 1] }, cuda_block!(2, 3));
    assert_eq!(super::ExecProp { grid_dim: [1, 1, 1], block_dim: [2, 1, 1] }, cuda_block!(2));
}

#[test]
fn cuda_ep_works() {
    assert_eq!(
        super::ExecProp { grid_dim: [2, 3, 4], block_dim: [1, 1, 1] },
        cuda_ep!(grid(2, 3, 4))
    );
    assert_eq!(
        super::ExecProp { grid_dim: [2, 3, 1], block_dim: [1, 1, 1] },
        cuda_ep!(grid(2, 3))
    );
    assert_eq!(
        super::ExecProp { grid_dim: [2, 1, 1], block_dim: [1, 1, 1] },
        cuda_ep!(grid(2))
    );

    assert_eq!(
        super::ExecProp { grid_dim: [1, 1, 1], block_dim: [2, 3, 4] },
        cuda_ep!(block(2, 3, 4))
    );
    assert_eq!(
        super::ExecProp { grid_dim: [1, 1, 1], block_dim: [2, 3, 1] },
        cuda_ep!(block(2, 3))
    );
    assert_eq!(
        super::ExecProp { grid_dim: [1, 1, 1], block_dim: [2, 1, 1] },
        cuda_ep!(block(2))
    );

    assert_eq!(
        super::ExecProp { grid_dim: [2, 3, 4], block_dim: [5, 6, 7] },
        cuda_ep!(grid(2, 3, 4), block(5, 6, 7))
    );
    assert_eq!(
        super::ExecProp { grid_dim: [2, 3, 4], block_dim: [5, 6, 7] },
        cuda_ep!(block(5, 6, 7), grid(2, 3, 4))
    );
}

}

