#[macro_use]
extern crate cuda_hlb;

cuda_module! {
    Adder {
        binary(include_bytes!("add.ptx"))
    }
}

#[test]
fn create_cuda_module() {
    let cuda = cuda_hlb::Cuda::with_primary_context().unwrap();
    let adder: Adder = cuda.load_module().unwrap();
}
