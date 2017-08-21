
use super::Result;
use super::ffi;
use super::init;
use std::mem::uninitialized;

pub struct Stream {
    pub(super) handle: ffi::CUstream,
}

impl Stream {
    pub fn new() -> Result<Stream> {
        init();

        unsafe {
            let mut handle = uninitialized();
            ffi::cuStreamCreate(&mut handle, ffi::CUstream_flags::CU_STREAM_DEFAULT as u32);
            Ok(Stream { handle })
        }
    }

    fn new_non_blocking() -> Result<Stream> {
        init();

        unsafe {
            let mut handle = uninitialized();
            ffi::cuStreamCreate(
                &mut handle,
                ffi::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            );
            Ok(Stream { handle })
        }
    }

    fn wait(&self) -> Result<()> {
        unsafe {
            ffi::cuStreamSynchronize(self.handle)?;
            Ok(())
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            ffi::cuStreamDestroy_v2(self.handle);
        }
    }
}
