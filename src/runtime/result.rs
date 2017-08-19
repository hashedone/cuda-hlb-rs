use ffi::runtime as ffi;
use std;
use std::mem::uninitialized;

#[derive(Clone, Copy, Debug)]
pub enum Error {
    InvalidValue,
    OutOfMemory,
    NotInitialized,
    Deinitialized,
    ProfilerDisabled,
    ProfilerNotInitialized,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    NoDevice,
    InvalidDevice,
    InvalidImage,
    InvalidContext,
    ContextAlreadyCurrent,
    MapFailed,
    UnmapFailed,
    ArrayIsMapped,
    AlreadyMapped,
    NoBinaryForGpu,
    AlreadyAcquired,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    EccUncorrectable,
    UnsupportedLimit,
    ContextAlreadyInUse,
    PeerAccessUnsupported,
    InvalidPtx,
    InvalidGraphicsContext,
    NvlinkUncorrectable,
    InvalidSource,
    FileNotFound,
    SharedObjectSymbolNotFound,
    SharedObjectInitFailed,
    OperatingSystem,
    InvalidHandle,
    NotFound,
    NotReady,
    IllegalAddress,
    LaunchOutOfResources,
    LaunchTimeout,
    LaunchIncompatibleTexturing,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    PrimaryContextActive,
    ContextIsDestroyed,
    Assert,
    TooManyPeers,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    HardwareStackError,
    IllegalInstruction,
    MisalignedAddress,
    InvalidAddressSpace,
    InvalidPc,
    LaunchFailed,
    NotPermitted,
    NotSupported,
    Unknown,
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::ops::Try for ffi::CUresult {
    type Ok = ();
    type Error = Error;

    fn into_result(self) -> Result<()> {
        match self {
            ffi::CUresult::CUDA_SUCCESS => Ok(()),
            ffi::CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue),
            ffi::CUresult::CUDA_ERROR_OUT_OF_MEMORY => Err(Error::OutOfMemory),
            ffi::CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized),
            ffi::CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized),
            ffi::CUresult::CUDA_ERROR_PROFILER_DISABLED => Err(Error::ProfilerDisabled),
            ffi::CUresult::CUDA_ERROR_PROFILER_NOT_INITIALIZED => Err(Error::ProfilerNotInitialized),
            ffi::CUresult::CUDA_ERROR_PROFILER_ALREADY_STARTED => Err(Error::ProfilerAlreadyStarted),
            ffi::CUresult::CUDA_ERROR_PROFILER_ALREADY_STOPPED => Err(Error::ProfilerAlreadyStopped),
            ffi::CUresult::CUDA_ERROR_NO_DEVICE => Err(Error::NoDevice),
            ffi::CUresult::CUDA_ERROR_INVALID_DEVICE => Err(Error::InvalidDevice),
            ffi::CUresult::CUDA_ERROR_INVALID_IMAGE => Err(Error::InvalidImage),
            ffi::CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext),
            ffi::CUresult::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => Err(Error::ContextAlreadyCurrent),
            ffi::CUresult::CUDA_ERROR_MAP_FAILED => Err(Error::MapFailed),
            ffi::CUresult::CUDA_ERROR_UNMAP_FAILED => Err(Error::UnmapFailed),
            ffi::CUresult::CUDA_ERROR_ARRAY_IS_MAPPED => Err(Error::ArrayIsMapped),
            ffi::CUresult::CUDA_ERROR_ALREADY_MAPPED => Err(Error::AlreadyMapped),
            ffi::CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU => Err(Error::NoBinaryForGpu),
            ffi::CUresult::CUDA_ERROR_ALREADY_ACQUIRED => Err(Error::AlreadyAcquired),
            ffi::CUresult::CUDA_ERROR_NOT_MAPPED => Err(Error::NotMapped),
            ffi::CUresult::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Err(Error::NotMappedAsArray),
            ffi::CUresult::CUDA_ERROR_NOT_MAPPED_AS_POINTER => Err(Error::NotMappedAsPointer),
            ffi::CUresult::CUDA_ERROR_ECC_UNCORRECTABLE => Err(Error::EccUncorrectable),
            ffi::CUresult::CUDA_ERROR_UNSUPPORTED_LIMIT => Err(Error::UnsupportedLimit),
            ffi::CUresult::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => Err(Error::ContextAlreadyInUse),
            ffi::CUresult::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => Err(Error::PeerAccessUnsupported),
            ffi::CUresult::CUDA_ERROR_INVALID_PTX => Err(Error::InvalidPtx),
            ffi::CUresult::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => Err(Error::InvalidGraphicsContext),
            ffi::CUresult::CUDA_ERROR_NVLINK_UNCORRECTABLE => Err(Error::NvlinkUncorrectable),
            ffi::CUresult::CUDA_ERROR_INVALID_SOURCE => Err(Error::InvalidSource),
            ffi::CUresult::CUDA_ERROR_FILE_NOT_FOUND => Err(Error::FileNotFound),
            ffi::CUresult::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => Err(Error::SharedObjectSymbolNotFound),
            ffi::CUresult::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => Err(Error::SharedObjectInitFailed),
            ffi::CUresult::CUDA_ERROR_OPERATING_SYSTEM => Err(Error::OperatingSystem),
            ffi::CUresult::CUDA_ERROR_INVALID_HANDLE => Err(Error::InvalidHandle),
            ffi::CUresult::CUDA_ERROR_NOT_FOUND => Err(Error::NotFound),
            ffi::CUresult::CUDA_ERROR_NOT_READY => Err(Error::NotReady),
            ffi::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS => Err(Error::IllegalAddress),
            ffi::CUresult::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => Err(Error::LaunchOutOfResources),
            ffi::CUresult::CUDA_ERROR_LAUNCH_TIMEOUT => Err(Error::LaunchTimeout),
            ffi::CUresult::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => Err(Error::LaunchIncompatibleTexturing),
            ffi::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => Err(Error::PeerAccessAlreadyEnabled),
            ffi::CUresult::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => Err(Error::PeerAccessNotEnabled),
            ffi::CUresult::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => Err(Error::PrimaryContextActive),
            ffi::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED => Err(Error::ContextIsDestroyed),
            ffi::CUresult::CUDA_ERROR_ASSERT => Err(Error::Assert),
            ffi::CUresult::CUDA_ERROR_TOO_MANY_PEERS => Err(Error::TooManyPeers),
            ffi::CUresult::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => Err(Error::HostMemoryAlreadyRegistered),
            ffi::CUresult::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => Err(Error::HostMemoryNotRegistered),
            ffi::CUresult::CUDA_ERROR_HARDWARE_STACK_ERROR => Err(Error::HardwareStackError),
            ffi::CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION => Err(Error::IllegalInstruction),
            ffi::CUresult::CUDA_ERROR_MISALIGNED_ADDRESS => Err(Error::MisalignedAddress),
            ffi::CUresult::CUDA_ERROR_INVALID_ADDRESS_SPACE => Err(Error::InvalidAddressSpace),
            ffi::CUresult::CUDA_ERROR_INVALID_PC => Err(Error::InvalidPc),
            ffi::CUresult::CUDA_ERROR_LAUNCH_FAILED => Err(Error::LaunchFailed),
            ffi::CUresult::CUDA_ERROR_NOT_PERMITTED => Err(Error::NotPermitted),
            ffi::CUresult::CUDA_ERROR_NOT_SUPPORTED => Err(Error::NotSupported),
            ffi::CUresult::CUDA_ERROR_UNKNOWN => Err(Error::Unknown)
        }
    }

    fn from_error(v: Error) -> ffi::CUresult {
        match v {
            Error::InvalidValue => ffi::CUresult::CUDA_ERROR_INVALID_VALUE,
            Error::OutOfMemory => ffi::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            Error::NotInitialized => ffi::CUresult::CUDA_ERROR_NOT_INITIALIZED,
            Error::Deinitialized => ffi::CUresult::CUDA_ERROR_DEINITIALIZED,
            Error::ProfilerDisabled => ffi::CUresult::CUDA_ERROR_PROFILER_DISABLED,
            Error::ProfilerNotInitialized => ffi::CUresult::CUDA_ERROR_PROFILER_NOT_INITIALIZED,
            Error::ProfilerAlreadyStarted => ffi::CUresult::CUDA_ERROR_PROFILER_ALREADY_STARTED,
            Error::ProfilerAlreadyStopped => ffi::CUresult::CUDA_ERROR_PROFILER_ALREADY_STOPPED,
            Error::NoDevice => ffi::CUresult::CUDA_ERROR_NO_DEVICE,
            Error::InvalidDevice => ffi::CUresult::CUDA_ERROR_INVALID_DEVICE,
            Error::InvalidImage => ffi::CUresult::CUDA_ERROR_INVALID_IMAGE,
            Error::InvalidContext => ffi::CUresult::CUDA_ERROR_INVALID_CONTEXT,
            Error::ContextAlreadyCurrent => ffi::CUresult::CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
            Error::MapFailed => ffi::CUresult::CUDA_ERROR_MAP_FAILED,
            Error::UnmapFailed => ffi::CUresult::CUDA_ERROR_UNMAP_FAILED,
            Error::ArrayIsMapped => ffi::CUresult::CUDA_ERROR_ARRAY_IS_MAPPED,
            Error::AlreadyMapped => ffi::CUresult::CUDA_ERROR_ALREADY_MAPPED,
            Error::NoBinaryForGpu => ffi::CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU,
            Error::AlreadyAcquired => ffi::CUresult::CUDA_ERROR_ALREADY_ACQUIRED,
            Error::NotMapped => ffi::CUresult::CUDA_ERROR_NOT_MAPPED,
            Error::NotMappedAsArray => ffi::CUresult::CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
            Error::NotMappedAsPointer => ffi::CUresult::CUDA_ERROR_NOT_MAPPED_AS_POINTER,
            Error::EccUncorrectable => ffi::CUresult::CUDA_ERROR_ECC_UNCORRECTABLE,
            Error::UnsupportedLimit => ffi::CUresult::CUDA_ERROR_UNSUPPORTED_LIMIT,
            Error::ContextAlreadyInUse => ffi::CUresult::CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
            Error::PeerAccessUnsupported => ffi::CUresult::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
            Error::InvalidPtx => ffi::CUresult::CUDA_ERROR_INVALID_PTX,
            Error::InvalidGraphicsContext => ffi::CUresult::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
            Error::NvlinkUncorrectable => ffi::CUresult::CUDA_ERROR_NVLINK_UNCORRECTABLE,
            Error::InvalidSource => ffi::CUresult::CUDA_ERROR_INVALID_SOURCE,
            Error::FileNotFound => ffi::CUresult::CUDA_ERROR_FILE_NOT_FOUND,
            Error::SharedObjectSymbolNotFound => ffi::CUresult::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
            Error::SharedObjectInitFailed => ffi::CUresult::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
            Error::OperatingSystem => ffi::CUresult::CUDA_ERROR_OPERATING_SYSTEM,
            Error::InvalidHandle => ffi::CUresult::CUDA_ERROR_INVALID_HANDLE,
            Error::NotFound => ffi::CUresult::CUDA_ERROR_NOT_FOUND,
            Error::NotReady => ffi::CUresult::CUDA_ERROR_NOT_READY,
            Error::IllegalAddress => ffi::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
            Error::LaunchOutOfResources => ffi::CUresult::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
            Error::LaunchTimeout => ffi::CUresult::CUDA_ERROR_LAUNCH_TIMEOUT,
            Error::LaunchIncompatibleTexturing => ffi::CUresult::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
            Error::PeerAccessAlreadyEnabled => ffi::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
            Error::PeerAccessNotEnabled => ffi::CUresult::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
            Error::PrimaryContextActive => ffi::CUresult::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
            Error::ContextIsDestroyed => ffi::CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
            Error::Assert => ffi::CUresult::CUDA_ERROR_ASSERT,
            Error::TooManyPeers => ffi::CUresult::CUDA_ERROR_TOO_MANY_PEERS,
            Error::HostMemoryAlreadyRegistered => ffi::CUresult::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
            Error::HostMemoryNotRegistered => ffi::CUresult::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
            Error::HardwareStackError => ffi::CUresult::CUDA_ERROR_HARDWARE_STACK_ERROR,
            Error::IllegalInstruction => ffi::CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION,
            Error::MisalignedAddress => ffi::CUresult::CUDA_ERROR_MISALIGNED_ADDRESS,
            Error::InvalidAddressSpace => ffi::CUresult::CUDA_ERROR_INVALID_ADDRESS_SPACE,
            Error::InvalidPc => ffi::CUresult::CUDA_ERROR_INVALID_PC,
            Error::LaunchFailed => ffi::CUresult::CUDA_ERROR_LAUNCH_FAILED,
            Error::NotPermitted => ffi::CUresult::CUDA_ERROR_NOT_PERMITTED,
            Error::NotSupported => ffi::CUresult::CUDA_ERROR_NOT_SUPPORTED,
            Error::Unknown => ffi::CUresult::CUDA_ERROR_UNKNOWN,
        }
    }

    fn from_ok(_: ()) -> Self {
        ffi::CUresult::CUDA_SUCCESS
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use std::ops::Try;
        let e = ffi::CUresult::from_error(self.clone());

        let s = unsafe {
            let mut buf = uninitialized();
            match ffi::cuGetErrorString(e, &mut buf) {
                ffi::CUresult::CUDA_SUCCESS => Ok(()),
                _ => Err(std::fmt::Error::default())
            }?;

            std::ffi::CStr::from_ptr(buf)
                .to_str()
                .map_err(|_| std::fmt::Error::default())?
        };

        write!(f, "{}", s)
    }
}
