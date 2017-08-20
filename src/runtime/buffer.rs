use super::ffi;
use super::Result;
use super::Context;
use super::Stream;
use std::marker::PhantomData;
use std::mem::size_of;
use std::mem::uninitialized;
use std::ops::Deref;
use std::ops::DerefMut;
use std;

pub struct BufferBuilder<'a, T> {
    context: &'a Context,
    phantom: PhantomData<T>
}

pub struct HostBuffer<'a, T> {
    context: &'a Context,
    ptr: *mut T,
    cnt:  usize
}

impl<'a, T: Sized+Copy> HostBuffer<'a, T> {
}

impl<'a, T> Drop for HostBuffer<'a, T> {
    fn drop(&mut self) {
        self.context.make_current().ok();
        unsafe {
            ffi::cuMemFreeHost(self.ptr as *mut _);
        }
    }
}

impl<'a, T> Deref for HostBuffer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.cnt) }
    }
}

impl<'a, T> DerefMut for HostBuffer<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.cnt) }
    }
}

pub struct Buffer<'a, T> {
    context: &'a Context,
    ptr: ffi::CUdeviceptr,
    cnt: usize,
    phantom: PhantomData<T>
}

impl<'a, T: Sized+Copy> Buffer<'a, T> {
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        if self.cnt != data.len() { return Err(super::result::Error::InvalidValue); }
        self.context.make_current()?;
        unsafe {
            ffi::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const _, data.len())?;
            Ok(())
        }
    }

    pub fn copy_from_host_async(&mut self, data: &[T], s: Stream) -> Result<()> {
        if self.cnt != data.len() { return Err(super::result::Error::InvalidValue); }
        self.context.make_current()?;
        unsafe {
            ffi::cuMemcpyHtoDAsync_v2(self.ptr, data.as_ptr() as *const _, data.len(), s.handle)?;
            Ok(())
        }
    }

    pub fn copy_to_host(&self, ptr: &mut [T]) -> Result<()> {
        if self.cnt != ptr.len() { return Err(super::result::Error::InvalidValue); }
        self.context.make_current()?;
        unsafe {
            ffi::cuMemcpyDtoH_v2(ptr.as_mut_ptr() as *mut _, self.ptr, ptr.len())?;
            Ok(())
        }
    }
    
    pub fn copy_to_host_async(&self, ptr: &mut [T], s: Stream) -> Result<()> {
        if self.cnt != ptr.len() { return Err(super::result::Error::InvalidValue); }
        self.context.make_current()?;
        unsafe {
            ffi::cuMemcpyDtoHAsync_v2(ptr.as_mut_ptr() as *mut _, self.ptr, ptr.len(), s.handle)?;
            Ok(())
        }
    }

}

impl<'a, T> Drop for Buffer<'a, T> {
    fn drop(&mut self) {
        self.context.make_current().ok();
        unsafe {
            ffi::cuMemFree_v2(self.ptr);
        }
    }
}

impl<'a, T: Sized+Copy> BufferBuilder<'a, T> {
    pub(super) fn new(context: &'a Context) -> BufferBuilder<'a, T> {
        BufferBuilder { context, phantom: PhantomData }
    }

    pub unsafe fn new_host(self, cnt: usize) -> Result<HostBuffer<'a, T>> {
        let mut ptr = uninitialized();
        self.context.make_current()?;
        ffi::cuMemAllocHost_v2(&mut ptr, size_of::<T>() * cnt)?;
        let ptr = ptr as *mut T;
        Ok(HostBuffer { context: self.context, ptr, cnt })
    }

    pub fn host_from_slice(self, data: &[T]) -> Result<HostBuffer<'a, T>> {
        let mut buf = unsafe { self.new_host(data.len())? };
        // context.make_current() may be ommited, cause its called in self.new_host
        buf.copy_from_slice(data);
        Ok(buf)
    }
    
    pub unsafe fn new_device(self, cnt: usize) -> Result<Buffer<'a, T>> {
        let mut ptr = uninitialized();
        self.context.make_current()?;
        ffi::cuMemAlloc_v2(&mut ptr, size_of::<T>() * cnt)?;
        Ok(Buffer { context: self.context, ptr, cnt, phantom: PhantomData })
    }

    pub fn device_from_slice(self, data: &[T]) -> Result<Buffer<'a, T>> {
        unsafe {
            let buf = self.new_device(data.len())?;
            // context.make_current() may be ommited, cause its called in self.new_device
            ffi::cuMemcpyHtoD_v2(buf.ptr, data.as_ptr() as *const _, data.len());
            Ok(buf)
        }
    }

    pub fn device_from_slice_async(self, data: &[T], s: &Stream) -> Result<Buffer<'a, T>> {
        unsafe {
            let buf = self.new_device(data.len())?;
            // context.make_current() may be ommited, cause its called in self.new_device
            ffi::cuMemcpyHtoDAsync_v2(buf.ptr, data.as_ptr() as *const _, data.len(), s.handle);
            Ok(buf)
        }
    }
}
