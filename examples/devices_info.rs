/*#![feature(catch_expr)]
extern crate cuda_hlb as cuda;

fn main() {
    do catch {
        println!("CUDA version: {}", cuda::version()?);
        cuda::init()?;
        for idx in 0..cuda::Device::count()? {
            let dev = cuda::Device::get(idx)?;
            println!("CUDA device {}: {}, total memory: {} bytes", idx, dev.name()?, dev.memory()?);
        }
        Ok(())
    }.unwrap_or_else(|e: cuda::result::CUError| println!("{}", e));
}*/
fn main() {}
