use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn get_python_site_dir() -> String {
    let site_dir_cmd = Command::new("python")
        .arg("-c")
        .arg("import site; print(site.getsitepackages()[0])")
        .stdout(Stdio::piped())
        .output()
        .expect("Failed to get python site packge dir.");

    let site_dir_full = String::from_utf8_lossy(&site_dir_cmd.stdout);
    let site_dir_path = site_dir_full.strip_suffix("\n").unwrap();

    site_dir_path.to_string()
}

fn main() {
    let python_site_dir = get_python_site_dir();
    let tensorflow_include = format!("{}/tensorflow/include", python_site_dir);
    let tensorflow_link = format!("{}/tensorflow", python_site_dir);
    let tensorflow_python_link = format!("{}/tensorflow/python", python_site_dir);

    println!("cargo:rustc-link-search={}", tensorflow_python_link);
    println!("cargo:rustc-link-search={}", tensorflow_link);

    println!("cargo:rustc-link-arg=-l:_pywrap_tensorflow_internal.so");
    println!("cargo:rustc-link-arg=-l:libtensorflow_framework.so.2");

    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}", tensorflow_include))
        .clang_arg(format!("-I{}", tensorflow_link))
        .clang_args(&["-x", "c++", "-std=c++17"])
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .blocklist_function("qfcvt_r")
        .blocklist_function("qecvt_r")
        .blocklist_function("qgcvt")
        .blocklist_function("qfcvt")
        .blocklist_function("qecvt")
        .blocklist_function("strtof64x_l")
        .blocklist_function("strtold_l")
        .blocklist_function("strfromf64x")
        .blocklist_function("strfroml")
        .blocklist_function("strtof64x")
        .blocklist_function("strtold")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
