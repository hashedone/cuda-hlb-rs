pub unsafe trait Module<'a>: Sized + 'a {
    fn load(cuda: &'a super::Cuda) -> super::Result<Self>;
}

#[macro_export]
macro_rules! cuda_module_gen {
    ($name:ident
        $load_module:expr
    ) => {
        use std as mstd;

        struct $name<'a> {
            cuda: &'a $crate::Cuda,
            module: $crate::ffi::CUmodule,
        }

        unsafe impl<'a> $crate::module::Module<'a> for $name<'a> {
            fn load(cuda: &'a $crate::Cuda) -> $crate::Result<$name<'a>> {
                cuda.make_current()?;
                let module = $load_module;
                Ok($name { cuda, module })
            }
        }

        impl<'a> Drop for $name<'a> {
            fn drop(&mut self) {
                self.cuda.make_current().ok();
                unsafe {
                    $crate::ffi::cuModuleUnload(self.module);
                }
            }
        }
    };
}

#[macro_export]
macro_rules! parse_module_loader {
    ($name:ident {
        binary($bin:expr);
    }) => {
        cuda_module_gen! {
            $name {
                let data: Vec<_> = $bin.iter().cloned().collect();
                let data = mstd::ffi::CString::new(data)?;
                unsafe {
                    let mut module = mstd::mem::uninitialized();
                    $crate::ffi::cuModuleLoadData(&mut module, data.as_ptr() as *const _)?;
                    module
                }
            }
        }
    };

    ($name:ident {
        binary_file($file:expr);
    }) => {
        cuda_module_gen! {
            $name {
                let file: String = $file.into();
                let file = mstd::ffi::CString::new(file)?;
                unsafe {
                    let mut module = mstd::mem::uninitialized();
                    $crate::ffi::cuModuleLoad(&mut module, file.as_ptr() as *const _)?;
                    module
                }
            }
        }
    };
}

#[macro_export]
macro_rules! cuda_module {
    ($($t:tt)*) => { parse_module_loader!($($t)*); };
}

#[cfg(test)]
mod test {

    cuda_module! {
    Adder {
        binary(include_bytes!("../tests/add.ptx"));
    }
}

    #[test]
    fn create_cuda_module() {
        let cuda = ::Cuda::with_primary_context().unwrap();
        let _: Adder = cuda.load_module().unwrap();
    }

}
