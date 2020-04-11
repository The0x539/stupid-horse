fn main() {
    for shader in [
        "vertex",
        "fragment",
    ].iter() {
        println!("cargo:rerun-if-changed=src/shaders/{}.glsl", shader);
    }
}
