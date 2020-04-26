fn main() {
    for shader in [
        "bg.vert",
        "bg.frag",
        "fg.vert",
        "fg.frag",
        "horse.vert",
        "horse.frag",
    ].iter() {
        println!("cargo:rerun-if-changed=src/shaders/{}", shader);
    }
}
