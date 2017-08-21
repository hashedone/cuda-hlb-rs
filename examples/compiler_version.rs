extern crate cuda_hlb as cuda;

#[cfg(feature = "compiler")]
fn main() {
    let (maj, min) = cuda::compiler::version().unwrap();
    println!("NVRTC version: {}.{}", maj, min);

    let program = r#"__global__ void hello(int *a, int *b)
        {
            a[threadIdx.x] += b[threadIdx.x];
        }"#;

    let prog = cuda::compiler::Compiler::new()
        .src(program)
        .add_name_expression("hello")
        .compile()
        .unwrap();

    println!("PTX size: {}", prog.ptx().unwrap().len());
    println!("Compilation log:\n{}", prog.log().unwrap());
    println!("Mangled kernel name:{}", prog.get_mangled("hello").unwrap());
}

#[cfg(not(feature = "compiler"))]
fn main() {}
