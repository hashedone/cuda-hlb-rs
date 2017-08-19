extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    #[cfg(feature="compiler")]
    {
        println!("cargo:rustc-link-lib=nvrtc");

        let bindings = bindgen::Builder::default()
            .header("compiler.h")
            .generate()
            .expect("Unable to generate compiler bindings");
        
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        
        bindings
            .write_to_file(out_path.join("bindings_compiler.rs"))
            .expect("Couln't write compiler bindings");
    }

    #[cfg(feature="runtime")]
    {
        println!("cargo:rustc-link-lib=cuda");
        let bindings = bindgen::Builder::default()
            .header("wrapper.h")
            .generate()
            .expect("Unable to generate device bindings");

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

        bindings
            .write_to_file(out_path.join("bindings_driver.rs"))
            .expect("Couldn't write driver bindings");
    }

}
