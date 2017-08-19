use super::ffi;
use super::Result;
use super::context::PrimaryContext;
use std::mem::uninitialized;
use libc;
use std;

#[derive(Clone, Copy)]
pub struct Device {
    handle: ffi::CUdevice,
}

pub struct Attributes {
    pub max_threads_per_block: i32,
    pub max_block_dim: [i32; 3]
}

impl Device {
    pub(super) fn new(handle: ffi::CUdevice) -> Device {
        Device { handle }
    }

    pub fn name(&self) -> Result<std::string::String> {
        let s = unsafe {
            let mut buf = [uninitialized(); 256];
            ffi::cuDeviceGetName(buf.as_mut_ptr(), buf.len() as libc::c_int, self.handle)?;
            std::ffi::CStr::from_ptr(buf.as_ptr())
                .to_str().unwrap()
                .to_string()
        };
        Ok(s)
    }

    pub fn memory(&self) -> Result<usize> {
        unsafe {
            let mut m = uninitialized();
            ffi::cuDeviceTotalMem_v2(&mut m, self.handle)?;
            Ok(m as usize)
        }
    } 

    pub fn pci_bus_id(&self) -> Result<String> {
        let s = unsafe {
            let mut buf = [uninitialized(); 13];
            ffi::cuDeviceGetPCIBusId(buf.as_mut_ptr(), 13, self.handle)?;
            std::ffi::CStr::from_ptr(buf.as_ptr())
                .to_str().unwrap()
                .to_string()
        };
        Ok(s)
    }

    // TODO: wrap cuDeviceGetAttribute
    pub fn attributes(&self) -> Result<Attributes> {
        unsafe {
            let mut attrs: Attributes = uninitialized();
            ffi::cuDeviceGetAttribute(
                &mut attrs.max_threads_per_block,
                ffi::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                self.handle)?;

            ffi::cuDeviceGetAttribute(
                &mut attrs.max_block_dim[0],
                ffi::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                self.handle)?;
            ffi::cuDeviceGetAttribute(
                &mut attrs.max_block_dim[1],
                ffi::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                self.handle)?;
            ffi::cuDeviceGetAttribute(
                &mut attrs.max_block_dim[2],
                ffi::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                self.handle)?;

            Ok(attrs)
        }
    }

    pub fn primary_context(&self) -> Result<PrimaryContext> {
        unsafe {
            let mut ctx = uninitialized();
            ffi::cuDevicePrimaryCtxRetain(&mut ctx, self.handle)?;
            Ok(PrimaryContext::new(self.handle, ctx))
        }
    }
}
