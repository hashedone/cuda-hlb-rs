use std;
use ffi::compiler as ffi;

#[derive(Clone, Copy, Debug)]
pub enum Error {
    OutOfMemory,
    ProgramCreationFailure,
    InvalidInput,
    InvalidProgram,
    InvalidOption,
    Compilation,
    BuiltinOperationFailure,
    NoNameExpressionsAfterCompilation,
    NoLoweredNamesBeforeCompilation,
    NameExpressionNotValid,
    InternalError
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::ops::Try for ffi::nvrtcResult {
    type Ok = ();
    type Error = Error;

    fn into_result(self) -> Result<()> {
        match self {
            ffi::nvrtcResult::NVRTC_SUCCESS => Ok(()),
            ffi::nvrtcResult::NVRTC_ERROR_OUT_OF_MEMORY =>
                Err(Error::OutOfMemory),
            ffi::nvrtcResult::NVRTC_ERROR_PROGRAM_CREATION_FAILURE =>
                Err(Error::ProgramCreationFailure),
            ffi::nvrtcResult::NVRTC_ERROR_INVALID_INPUT =>
                Err(Error::InvalidInput),
            ffi::nvrtcResult::NVRTC_ERROR_INVALID_PROGRAM =>
                Err(Error::InvalidProgram),
            ffi::nvrtcResult::NVRTC_ERROR_INVALID_OPTION =>
                Err(Error::InvalidOption),
            ffi::nvrtcResult::NVRTC_ERROR_COMPILATION =>
                Err(Error::Compilation),
            ffi::nvrtcResult::NVRTC_ERROR_BUILTIN_OPERATION_FAILURE =>
                Err(Error::BuiltinOperationFailure),
            ffi::nvrtcResult::NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION =>
                Err(Error::NoNameExpressionsAfterCompilation),
            ffi::nvrtcResult::NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION =>
                Err(Error::NoLoweredNamesBeforeCompilation),
            ffi::nvrtcResult::NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID =>
                Err(Error::NameExpressionNotValid),
            ffi::nvrtcResult::NVRTC_ERROR_INTERNAL_ERROR =>
                Err(Error::InternalError)
        }
    }

    fn from_error(v: Error) -> Self {
        match v {
            Error::OutOfMemory => 
                ffi::nvrtcResult::NVRTC_ERROR_OUT_OF_MEMORY,
            Error::ProgramCreationFailure => 
                ffi::nvrtcResult::NVRTC_ERROR_PROGRAM_CREATION_FAILURE,
            Error::InvalidInput => 
                ffi::nvrtcResult::NVRTC_ERROR_INVALID_INPUT,
            Error::InvalidProgram => 
                ffi::nvrtcResult::NVRTC_ERROR_INVALID_PROGRAM,
            Error::InvalidOption => 
                ffi::nvrtcResult::NVRTC_ERROR_INVALID_OPTION,
            Error::Compilation => 
                ffi::nvrtcResult::NVRTC_ERROR_COMPILATION,
            Error::BuiltinOperationFailure => 
                ffi::nvrtcResult::NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
            Error::NoNameExpressionsAfterCompilation => 
                ffi::nvrtcResult::NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
            Error::NoLoweredNamesBeforeCompilation => 
                ffi::nvrtcResult::NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
            Error::NameExpressionNotValid => 
                ffi::nvrtcResult::NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
            Error::InternalError => 
                ffi::nvrtcResult::NVRTC_ERROR_INTERNAL_ERROR,
        }
    }

    fn from_ok(_: ()) -> Self { ffi::nvrtcResult::NVRTC_SUCCESS }
}

