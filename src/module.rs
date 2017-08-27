pub unsafe trait Module<'a>: Sized + 'a {
    fn load(cuda: &'a super::Cuda) -> super::Result<Self>;
}

#[macro_export]
macro_rules! cuda_module_gen {
    ($name:ident,
     $load_module:expr,
     ($($fname:ident),*)
    ) => { interpolate_idents! {
        use std as mstd;

        struct $name<'a> {
            cuda: &'a $crate::Cuda,
            module: $crate::ffi::CUmodule,
            $(
                [fhandle_ $fname]: $crate::ffi::CUfunction,
            )*
        }

        unsafe impl<'a> $crate::module::Module<'a> for $name<'a> {
            fn load(cuda: &'a $crate::Cuda) -> $crate::Result<$name<'a>> {
                cuda.make_current()?;
                let module = $load_module;
                $(
                    let [fhandle_ $fname] = unsafe {
                        let mut handle = mstd::mem::uninitialized();
                        let name = mstd::ffi::CString::new(stringify!($fname))?;
                        $crate::ffi::cuModuleGetFunction(&mut handle, module, name.as_ptr() as *const _)?;
                        handle
                    };
                )*;
                Ok($name { cuda, module, $([fhandle_ $fname]),* })
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
    }};
}

#[macro_export]
macro_rules! parse_functions {
    ($name:ident,
     $load_module:expr,
     $(fn $fname:ident ($($aname:ident : $atype:ty),*);)*) => {
         cuda_module_gen! {
             $name,
             $load_module,
             ($($fname),*)
         }
     };
}

#[macro_export]
macro_rules! parse_module_loader {
    ($name:ident {
        binary($bin:expr);
        $($t:tt)*
    }) => {
        parse_functions! {
            $name, {
                let data: Vec<_> = $bin.iter().cloned().collect();
                let data = mstd::ffi::CString::new(data)?;
                unsafe {
                    let mut module = mstd::mem::uninitialized();
                    $crate::ffi::cuModuleLoadData(&mut module, data.as_ptr() as *const _)?;
                    module
                }
            }, $($t)*
        }
    };

    ($name:ident {
        binary_file($file:expr);
        $($t:tt)*
    }) => {
        parse_functions! {
            $name, {
                let file: String = $file.into();
                let file = mstd::ffi::CString::new(file)?;
                unsafe {
                    let mut module = mstd::mem::uninitialized();
                    $crate::ffi::cuModuleLoad(&mut module, file.as_ptr() as *const _)?;
                    module
                }
            }, $($t)*
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
            fn add(a: *mut u8, b: *mut u8);
        }
    }

    #[test]
    fn create_cuda_module() {
        let cuda = ::Cuda::with_primary_context().unwrap();
        let _: Adder = cuda.load_module().unwrap();
    }

}

