extern crate cuda_hlb as cuda;

fn main() {}

#[test]
#[cfg(feature="compiler")]
#[cfg(feature="runtime")]
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
    let prog = ctx.program_builder().from_compiled(prog).unwrap();
}
