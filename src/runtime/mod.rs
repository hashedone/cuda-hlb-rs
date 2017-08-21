pub mod result;
pub mod device;
pub mod context;
pub mod program;
pub mod buffer;
pub mod stream;

pub use self::buffer::Buffer;
pub use self::context::Context;
pub use self::device::Device;
pub use self::program::Program;

pub use self::result::Result;
pub use self::stream::Stream;
use ffi::runtime as ffi;
use std;
use std::mem::uninitialized;

#[derive(Clone, Copy)]
pub struct Cuda {}
static INIT: std::sync::Once = std::sync::ONCE_INIT;

pub fn init() {
    unsafe { INIT.call_once(|| { ffi::cuInit(0); }) };
}

pub fn version() -> Result<i32> {
    unsafe {
        let mut v = uninitialized();
        ffi::cuDriverGetVersion(&mut v)?;
        Ok(v)
    }
}

pub fn devices() -> Result<impl Iterator<Item = Device>> {
    init();

    let c = unsafe {
        let mut c = uninitialized();
        ffi::cuDeviceGetCount(&mut c)?;
        c
    };

    let res = (0..c)
        .map(|i| unsafe {
            let mut h = uninitialized();
            let r = ffi::cuDeviceGet(&mut h, i);
            match r {
                ffi::CUresult::CUDA_SUCCESS => Some(Device::new(h)),
                _ => None,
            }
        })
        .filter_map(|x| x);

    Ok(res)
}

pub fn device_by_bus_id<S: Into<String>>(bus: S) -> Result<Device> {
    init();

    unsafe {
        let bus = std::ffi::CString::new(bus.into()).unwrap();
        let mut h = uninitialized();
        ffi::cuDeviceGetByPCIBusId(&mut h, bus.as_ptr())?;
        Ok(Device::new(h))
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn get_version() {
        super::version().unwrap();
    }

    #[test]
    fn get_devices() {
        assert!(super::devices().unwrap().count() > 0);
    }

    #[test]
    fn device_props() {
        let dev = super::devices().unwrap().next().unwrap();
        assert!(dev.memory().unwrap() > 0);
        assert!(dev.name().unwrap().len() > 0);
        assert!(dev.pci_bus_id().unwrap().len() > 0);
    }
}
