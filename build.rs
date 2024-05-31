use std::process::{Command, Stdio};

fn get_python_site_dir() -> String {
    let site_dir_cmd = Command::new("python")
        .arg("-c")
        .arg("import site; print(site.getsitepackages()[0])")
        .stdout(Stdio::piped())
        .output()
        .expect("Failed to get python site packge dir.");

    let site_dir_full = String::from_utf8_lossy(&site_dir_cmd.stdout);
    let site_dir_path = site_dir_full.strip_suffix('\n').unwrap();

    site_dir_path.to_string()
}

fn main() {
    let python_site_dir = get_python_site_dir();
    let tensorflow_link = format!("{}/tensorflow", python_site_dir);
    let tensorflow_python_link = format!("{}/tensorflow/python", python_site_dir);

    println!("cargo:rustc-link-search={}", tensorflow_python_link);
    println!("cargo:rustc-link-search={}", tensorflow_link);

    println!("cargo:rustc-link-arg=-l:libtensorflow_framework.so.2");
    println!("cargo:rustc-link-arg=-l:_pywrap_tensorflow_internal.so");
}
