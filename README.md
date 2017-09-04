# cuda-hlb-rs

Crate providing high level interface for nVidia CUDA.

One of the best thing in nVidia CUDA is that its possible to almost transparently implement GPU computation.
However it is developed to be integrated with C++, so to use it with Rust, CUDA program has to be compiled to
PTX or FATBIN format, and then Driver API call has to be performed to manage it. Good news is, that Rust can
be compiled to PTX: [https://github.com/japaric/nvptx], so it may be possible to do integration of host CODE
and GPU code nearly as transparent, as in C++.

First goal of crate is to provide interface for loading external PTX files, but allowing almost transparent call
to functions defined there. This is more or less done already, but the interface may change - I'm not happy with
exposing all crate internals to user just because everything is macro generated and need to be public. Now
I am working on handling all possible CUDA types, in particular textures and surfaces.

When the Rust side API would support most of CUDA driver functionality, I would work on compiling it on stable
Rust. For now I am using some unstable features to make my work easier, but it would be nice, if crate would
not need them.

Last step would be somehow integrate GPU side and Host side code together, writting both in Rust. I expect
it could be done with procedural macros (https://github.com/rust-lang/rust/issues/38356), but for now its
in development phase. If there would be no update on this, when I would achieve this stage, I am planning to
try implementing this with current unstable compiler plugin API, but as long as it's highly unstable and
its future is unclear reffering to procedural macros and macros 2.0, I really want to aviod this try.

For now have fun, and check crate tests to check crate use cases. Also if what you need is just CUDA ffi bindings
with minimal wrappers check [https://github.com/japaric/cuda].
