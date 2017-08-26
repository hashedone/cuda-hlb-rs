#![allow(unused_doc_comment)]

use ffi;
use std;
use std::mem::uninitialized;

pub type DriverError = ffi::CUresult;

fn get_cuda_err_string(err: DriverError) -> String {
    unsafe {
        let mut buf = uninitialized();
        match ffi::cuGetErrorString(err, &mut buf) {
            ffi::CUresult::CUDA_SUCCESS => std::ffi::CStr::from_ptr(buf)
                .to_str()
                .map(|s| String::from(s))
                .unwrap_or_else(|_| String::new()),
            _ => String::new(),
        }
    }
}

error_chain! {
    foreign_links {
        CStringParse(std::ffi::NulError);
    }

    errors {
        Driver(e: DriverError) {
            display("Driver error: {}", get_cuda_err_string(*e))
        }
        NoDevices
        BuffersSizeMissmatch(source: usize, dest: usize) {
            display("Missmatching buffers size, source: {}, dest: {}", source, dest)
        }
    }
}

impl std::ops::Try for ffi::CUresult {
    type Ok = ();
    type Error = Error;

    fn into_result(self) -> Result<()> {
        match self {
            ffi::CUresult::CUDA_SUCCESS => Ok(()),
            err => Err(ErrorKind::Driver(err).into()),
        }
    }

    fn from_error(v: Error) -> ffi::CUresult {
        match v {
            Error(ErrorKind::Driver(err), _) => err,
            _ => ffi::CUresult::CUDA_ERROR_UNKNOWN,
        }
    }

    fn from_ok(_: ()) -> ffi::CUresult {
        ffi::CUresult::CUDA_SUCCESS
    }
}
