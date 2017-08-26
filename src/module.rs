pub unsafe trait Module<'a>: Sized+'a {
    unsafe fn load(cuda: &'a super::Cuda) -> super::Result<Self>;
}

#[macro_export]
macro_rules! cuda_module {
    ($name:ident {
        binary($bin:expr)
    }) => {
        use std as mstd;

        struct $name<'a> {
            cuda: &'a $crate::Cuda,
            module: $crate::ffi::CUmodule,
        }

        unsafe impl<'a> $crate::module::Module<'a> for $name<'a> {
            unsafe fn load(cuda: &'a $crate::Cuda) -> $crate::Result<$name<'a>> {
                let data = $bin.iter().cloned().collect::<Vec<_>>();
                let data = mstd::ffi::CString::new(data)?;
                let mut module = mstd::mem::uninitialized();
                $crate::ffi::cuModuleLoadData(&mut module, data.as_ptr() as *const _)?;
                Ok($name { cuda, module })
            }
        }

        impl<'a> mstd::ops::Drop for $name<'a> {
            fn drop(&mut self) {
                self.cuda.make_current().ok();
                unsafe {
                    $crate::ffi::cuModuleUnload(self.module);
                }
            }
        }
    };
}

mod test {

cuda_module! {
    Adder {
        binary(include_bytes!("../tests/add.ptx"))
    }
}

#[test]
fn create_cuda_module() {
    let cuda = ::Cuda::with_primary_context().unwrap();
    let adder = cuda.load_module::<Adder>();
}

}

