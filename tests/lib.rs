#[macro_use]
extern crate cuda_hlb;

cuda_mod! { ptx(add) {
    fn add(a: *mut u8, b: *mut u8);
} }

#[test]
fn simple_vectors_adding() {
    let cuda = cuda_hlb::Cuda::with_primary_context().unwrap();
    let adder: add::Module = cuda.load_module().unwrap();
    let bufa = cuda_buffer![cuda => 1u8; 100];
    let bufb = cuda_buffer![cuda => 2u8; 100];
    adder.add(&cuda_block![100], &bufa, &bufb).unwrap();
    assert_eq!(vec![3u8; 100], bufa.get_vec().unwrap());
}
