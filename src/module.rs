#[macro_export]
macro_rules! cuda_module {
    ($name:ident {
        binary($bin:expr)
    }) => {

        struct $name<'a> {
            cuda: &'a $crate::Cuda,
            module: $crate::ffi::CUmodule,
        }

        impl<'a> $name<'a> {
            fn new(cuda: &'a $crate::Cuda) -> $crate::Result<$name<'a>> {
                cuda.make_current()?;
                let data = $bin.iter().cloned().collect::<Vec<_>>();
                let data = std::ffi::CString::new(data)?;
                unsafe {
                    let mut module = std::mem::uninitialized();
                    $crate::ffi::cuModuleLoadData(&mut module, data.as_ptr() as *const _)?;
                    Ok($name { cuda, module })
                }
            }
        }

        impl<'a> std::ops::Drop for $name<'a> {
            fn drop(&mut self) {
                self.cuda.make_current().ok();
                unsafe {
                    $crate::ffi::cuModuleUnload(self.module);
                }
            }
        }
    };
}
