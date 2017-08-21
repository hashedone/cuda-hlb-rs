extern crate cuda_hlb as cuda;

fn main() {}

#[test]
#[cfg(feature = "compiler")]
#[cfg(feature = "runtime")]
fn load_kernel() {
    let program = r#"__global__ void hello(int *a, int *b)
    {
        a[threadIdx.x] += b[threadIdx.x];
    }"#;

    let prog = cuda::compiler::Compiler::new()
        .src(program)
        .compile()
        .unwrap();

    let dev = cuda::runtime::devices().unwrap().next().unwrap();
    let ctx = dev.primary_context().unwrap();
    let prog = ctx.program_builder().build_from_compiled(prog).unwrap();
}

#[test]
#[cfg(feature = "compiler")]
#[cfg(feature = "runtime")]
fn simple_adding() {
    let program = r#"__global__ void add(int *a, int *b)
    {
        a[threadIdx.x] += b[threadIdx.x];
    }"#;

    let prog = cuda::compiler::Compiler::new()
        .src(program)
        .add_name_expression("add")
        .compile()
        .unwrap();
    let hello_mangled = prog.get_mangled("add").unwrap();
    let dev = cuda::runtime::devices().unwrap().next().unwrap();
    let ctx = dev.primary_context().unwrap();
    let prog = ctx.program_builder().build_from_compiled(prog).unwrap();
    let kernel = prog.get_kernel(hello_mangled);
    let s = cuda::runtime::Stream::new().unwrap();

    let a = ctx.buffer_builder().device_from_slice(&[1 as i32; 128]);
    let b = ctx.buffer_builder().device_from_slice(&[2 as i32; 128]);
}
