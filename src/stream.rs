use super::result::Result;
use ffi;
use std::marker::PhantomData;
use std::mem::uninitialized;

pub struct Stream<'a> {
    stream: ffi::CUstream,
    phantom: PhantomData<&'a super::Cuda>,
}

impl<'a> Stream<'a> {
    pub(super) fn new() -> Result<Stream<'a>> {
        unsafe {
            let mut stream = uninitialized();
            ffi::cuStreamCreate(&mut stream, ffi::CUstream_flags::CU_STREAM_DEFAULT as u32)?;
            Ok(Stream {
                stream,
                phantom: PhantomData,
            })
        }
    }

    pub(super) fn new_non_blocking() -> Result<Stream<'a>> {
        unsafe {
            let mut stream = uninitialized();
            ffi::cuStreamCreate(
                &mut stream,
                ffi::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            )?;
            Ok(Stream {
                stream,
                phantom: PhantomData,
            })
        }
    }
}

impl<'a> Drop for Stream<'a> {
    fn drop(&mut self) {
        unsafe {
            ffi::cuStreamDestroy_v2(self.stream);
        }
    }
}
