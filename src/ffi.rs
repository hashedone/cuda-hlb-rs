#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]

// include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(feature = "compiler")]
pub mod compiler {
    include!(concat!(env!("OUT_DIR"), "/bindings_compiler.rs"));
}

#[cfg(feature = "runtime")]
pub mod runtime {
    include!(concat!(env!("OUT_DIR"), "/bindings_driver.rs"));
}
