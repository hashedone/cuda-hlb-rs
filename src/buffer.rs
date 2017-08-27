use super::Cuda;
use super::Result;
use super::ffi;
use super::result;
use std::collections::Bound;
use std::collections::range::RangeArgument;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::mem::{size_of, uninitialized};
use std::ops::Deref;

#[derive(Clone)]
pub struct BufferView<'a, T> {
    cuda: &'a Cuda,
    ptr: ffi::CUdeviceptr,
    size: usize,
    phantom: PhantomData<T>,
}

pub struct Buffer<'a, T> {
    buffer: BufferView<'a, T>,
}

impl<'a, T: Sized> Buffer<'a, T> {
    pub(super) unsafe fn uninitialized(cuda: &'a Cuda, cnt: usize) -> Result<Buffer<'a, T>> {
        let size = size_of::<T>() * cnt;
        cuda.make_current()?;
        let mut ptr = uninitialized();
        ffi::cuMemAlloc_v2(&mut ptr, size as _);
        Ok(Buffer {
            buffer: BufferView {
                cuda,
                ptr,
                size: cnt,
                phantom: PhantomData,
            },
        })
    }

    pub(super) fn new(cuda: &'a Cuda, cnt: usize) -> Result<Buffer<'a, T>> {
        let buf = unsafe { Self::uninitialized(cuda, cnt)? };

        let size = size_of::<T>() * cnt;
        let alignment = size % 4;
        match alignment {
            0 => unsafe {
                ffi::cuMemsetD32_v2(buf.buffer.ptr, 0, (size / 4) as _)?;
            },
            1 | 3 => unsafe {
                ffi::cuMemsetD8_v2(buf.buffer.ptr, 0, size as _)?;
            },
            2 => unsafe {
                ffi::cuMemsetD16_v2(buf.buffer.ptr, 0, size as _)?;
            },
            _ => unreachable!(),
        };

        Ok(buf)
    }
}

impl<'a, T> Drop for Buffer<'a, T> {
    fn drop(&mut self) {
        self.buffer.cuda.make_current().ok();
        unsafe {
            ffi::cuMemFree_v2(self.buffer.ptr);
        }
    }
}

impl<'a, T> Deref for Buffer<'a, T> {
    type Target = BufferView<'a, T>;

    fn deref(&self) -> &BufferView<'a, T> {
        &self.buffer
    }
}

impl<'a, T: Sized + Copy> BufferView<'a, T> {
    pub fn len(&self) -> usize {
        self.size
    }
    
    pub fn is_empty(&self) -> bool { self.size == 0 }

    pub fn load_slice(&self, data: &[T]) -> Result<()> {
        if data.len() != self.size {
            bail!(result::ErrorKind::BuffersSizeMissmatch(
                data.len(),
                self.size
            ));
        }

        if self.size == 0 { return Ok(()); }

        self.cuda.make_current()?;
        unsafe {
            ffi::cuMemcpyHtoDAsync_v2(
                self.ptr,
                data.as_ptr() as *const _,
                data.len() * size_of::<T>(),
                0 as _,
            )?;
        }
        Ok(())
    }

    pub fn write_to_slice(&self, dest: &mut [T]) -> Result<()> {
        if dest.len() != self.size {
            bail!(result::ErrorKind::BuffersSizeMissmatch(
                self.size,
                dest.len()
            ));
        }

        if self.size == 0 { return Ok(()); }

        self.cuda.make_current()?;
        unsafe {
            ffi::cuMemcpyDtoHAsync_v2(
                dest.as_mut_ptr() as *mut _,
                self.ptr,
                dest.len() * size_of::<T>(),
                0 as _,
            )?;
        }
        Ok(())
    }

    pub fn get_vec(&self) -> Result<Vec<T>> {
        if self.size == 0 { return Ok(vec![]); }

        unsafe {
            let mut v = vec![uninitialized(); self.size];
            self.write_to_slice(&mut v)?;
            Ok(v)
        }
    }

    pub fn sub<R: RangeArgument<usize>>(&self, range: R) -> Result<BufferView<T>> {
        let start = match range.start() {
            Bound::Included(s) => *s,
            Bound::Excluded(s) => s + 1,
            Bound::Unbounded => 0,
        };
        if start >= self.size {
            bail!(result::ErrorKind::OutOfBound(start, self.size));
        }

        let end = match range.end() {
            Bound::Included(s) => s + 1,
            Bound::Excluded(s) => *s,
            Bound::Unbounded => self.size,
        };
        if end > self.size {
            bail!(result::ErrorKind::OutOfBound(start, self.size));
        }

        Ok(BufferView {
            cuda: self.cuda,
            ptr: self.ptr + (start * size_of::<T>()) as ffi::CUdeviceptr,
            size: end - start,
            phantom: PhantomData,
        })
    }
}

impl<'a, T: Copy> TryInto<Vec<T>> for BufferView<'a, T> {
    type Error = result::Error;

    fn try_into(self) -> Result<Vec<T>> {
        self.get_vec()
    }
}

// TODO: Use Box<[]> instead of Vec
// TODO: Use heap allocated array for small data
// TODO: Use CUDA host allocated array and profile results
// TODO: Use memset for cuda_buffer![cuda => val; cnt] syntax
//       if size_of::<T>() < 4 and T is 1/2/4 bytes aligned

#[macro_export]
macro_rules! cuda_buffer {
    ($cuda:expr => $elem:expr ; $n:expr) => { $cuda.buffer_from_slice(&vec![$elem;$n]).unwrap() };
    ($cuda:expr => $($x:expr) , *) => { $cuda.buffer_from_slice(&vec![$($x) , *]).unwrap() };
    ($cuda:expr => $($x:expr ,)*) => { $cuda.buffer_from_slice(&vec![$($x) , *]).unwrap() };
}

#[cfg(test)]
mod test {
    #[test]
    fn fill_buffer_fails_with_missmached_sizes() {
        let cuda = ::Cuda::with_primary_context().unwrap();
        let buffer = cuda.new_buffer(100).unwrap();
        {
            let data = vec![2; 99];
            match buffer.load_slice(&data) {
                Err(
                    super::result::Error(
                        super::result::ErrorKind::BuffersSizeMissmatch(99, 100),
                        _,
                    ),
                ) => (),
                e => panic!("Sizes should be missmached, but result is: {:?}", e),
            }
        }
        {
            let data = vec![2; 101];
            match buffer.load_slice(&data) {
                Err(
                    super::result::Error(
                        super::result::ErrorKind::BuffersSizeMissmatch(101, 100),
                        _,
                    ),
                ) => (),
                e => panic!("Sizes should be missmached, but result is: {:?}", e),
            }
        }
    }

    #[test]
    fn read_buffer_fails_with_missmached_sizes() {
        let cuda = ::Cuda::with_primary_context().unwrap();
        let buffer = cuda.new_buffer(100).unwrap();
        {
            let mut data = vec![2; 99];
            match buffer.write_to_slice(&mut data) {
                Err(
                    super::result::Error(
                        super::result::ErrorKind::BuffersSizeMissmatch(100, 99),
                        _,
                    ),
                ) => (),
                e => panic!("Sizes should be missmached, but result is: {:?}", e),
            }
        }
        {
            let mut data = vec![2; 101];
            match buffer.write_to_slice(&mut data) {
                Err(
                    super::result::Error(
                        super::result::ErrorKind::BuffersSizeMissmatch(100, 101),
                        _,
                    ),
                ) => (),
                e => panic!("Sizes should be missmached, but result is: {:?}", e),
            }
        }
    }

    #[test]
    fn buffer_from_slice() {
        let cuda = ::Cuda::with_primary_context().unwrap();
        let input = vec![2; 10];
        let buffer = cuda.buffer_from_slice(&input).unwrap();
        let output = buffer.get_vec().unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn load_then_read_buffer() {
        let cuda = ::Cuda::with_primary_context().unwrap();
        let buffer = cuda.new_buffer(10).unwrap();
        let input = vec![2; 10];
        buffer.load_slice(&input).unwrap();
        let output = buffer.get_vec().unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn buffer_subindexing() {
        let cuda = ::Cuda::with_primary_context().unwrap();
        let buffer = cuda.buffer_from_slice(&(0..10).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(
            (0..10).collect::<Vec<_>>(),
            buffer.sub(..).unwrap().get_vec().unwrap()
        );
        assert_eq!(
            (0..3).collect::<Vec<_>>(),
            buffer.sub(..3).unwrap().get_vec().unwrap()
        );
        assert_eq!(
            (4..7).collect::<Vec<_>>(),
            buffer.sub(4..7).unwrap().get_vec().unwrap()
        );
        assert_eq!(
            (8..10).collect::<Vec<_>>(),
            buffer.sub(8..).unwrap().get_vec().unwrap()
        );
    }

    #[test]
    fn filling_subrange() {
        let cuda = ::Cuda::with_primary_context().unwrap();
        let buffer = cuda.buffer_from_slice(&[0; 10]).unwrap();
        buffer.sub(3..6).unwrap().load_slice(&[2; 3]).unwrap();
        assert_eq!(
            vec![0, 0, 0, 2, 2, 2, 0, 0, 0, 0],
            buffer.get_vec().unwrap()
        );
    }

    #[test]
    fn cuda_buffer_macro() {
        let cuda = ::Cuda::with_primary_context().unwrap();
        let buffer = cuda_buffer!(cuda => 2;10);
        assert_eq!(vec![2; 10], buffer.get_vec().unwrap());
        let buffer = cuda_buffer!(cuda => 1,2,3);
        assert_eq!(vec![1, 2, 3], buffer.get_vec().unwrap());
        let buffer = cuda_buffer!(cuda => 1,2,3,);
        assert_eq!(vec![1, 2, 3], buffer.get_vec().unwrap());
    }
}
