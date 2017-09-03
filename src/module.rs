pub unsafe trait Module<'a>: Sized + 'a {
    fn load(cuda: &'a super::Cuda) -> super::Result<Self>;
}

#[macro_export]
macro_rules! cuda_mod {
    (ptx($name:ident: $src:expr) {
        $($tail:tt)*
    }) => {
        cuda_mod_expand! {
            $name {
                let data = include_bytes!($src);
                let data = [&data[..], &[0]].concat();
                unsafe {
                    let mut module = std::mem::uninitialized();
                    $crate::ffi::cuModuleLoadData(&mut module, data.as_ptr() as *const  _)?;
                    module
                }
            } => {
                $($tail)*
            }
        }
    };

    (ptx($name:ident) {
        $($tail:tt)*
    }) => {
        cuda_mod! {
            ptx($name: concat!(stringify!($name), ".ptx")) {
                $($tail)*
            }
        }
    };
}

#[macro_export]
macro_rules! cuda_mod_expand {
    ($name:ident $module_load:expr => {
        $(fn $fname:ident ($($farg:ident : $fargt:ty),*);)*
    }) => {
        #[allow(dead_code)]
        mod $name {
            use std;

            pub struct Handles {
                $($fname: $crate::ffi::CUfunction,)*
            }
            
            pub struct Module<'a> {
                cuda: &'a $crate::Cuda,
                module: $crate::ffi::CUmodule,
                handles: Handles,
            }

            unsafe impl<'a> $crate::module::Module<'a> for Module<'a> {
                fn load(cuda: &'a $crate::Cuda) -> $crate::Result<Module<'a>> {
                    cuda.make_current()?;
                    let module = $module_load;
                    let handles = Handles {
                        $($fname: unsafe {
                            let mut handle = std::mem::uninitialized();
                            let name = std::ffi::CString::new(stringify!($fname))?;
                            $crate::ffi::cuModuleGetFunction(&mut handle, module, name.as_ptr() as *const _)?;
                            handle
                        }),*
                    };
                    Ok(Module { cuda, module, handles })
                }
            }

            impl<'a> Drop for Module<'a> {
                fn drop(&mut self) {
                    self.cuda.make_current().ok();
                    unsafe { $crate::ffi::cuModuleUnload(self.module); } }
            }

            impl<'a> Module<'a> {
                $(gen_cuda_fn!{fn $fname($($farg : $fargt),*)})*
            }
        }
    };
}

#[macro_export]
macro_rules! gen_cuda_fn {
    (fn $fname:ident ($($farg:ident : $fargt:ty),*)) => {
        #[allow(non_camel_case_types)]
        pub fn $fname<$($farg: $crate::AsCudaType<$fargt>),*>(
            &self,
            ep: &$crate::ExecProp,
            $($farg : &$farg),*
        ) -> $crate::Result<()> {
            let args = [$(
                unsafe { $farg.cuda_type() as *const _ }
            ),*];
            
            self.cuda.make_current()?;
            unsafe {
                $crate::ffi::cuLaunchKernel(
                    self.handles.$fname,
                    ep.grid_dim[0] as u32, ep.grid_dim[1] as u32, ep.grid_dim[2] as u32,
                    ep.block_dim[0] as u32, ep.block_dim[1] as u32, ep.block_dim[2] as u32,
                    0, 0 as $crate::ffi::CUstream,
                    args.as_ptr() as *mut _,
                    0 as *mut _)?;
            }

            Ok(())
        }
    };
}

#[cfg(test)]
mod test {
    cuda_mod! { ptx(add: "../tests/add.ptx") {
        fn add(a: *mut u8, b: *mut u8);
    } }

    #[test]
    fn create_cuda_module() {
        let cuda = ::Cuda::with_primary_context().unwrap();
        let _: add::Module = cuda.load_module().unwrap();
    }

} 

