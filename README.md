# VultenRs
This is a Rust rewrite of [Vulten](https://github.com/jswag180/Vulten).

This SHOULD work with any Vulkan 1.3 compliant device on Linux.

# Building
Create a python venv with Tensorflow
```
python -m venv venv
. ./venv/bin/activate
pip install Tensorflow
```
Then
```
cargo build
```
or
```
cargo build --release
```

# Usage
For now there is no packaging mechanism so to load the plugin
```
TF_PLUGGABLE_DEVICE_LIBRARY_PATH=target/debug/libvulten_rs.so python3 test_tf.py
```
Vulten devices use the VULK name so to use them
```
with tf.device("VULK:0"):
    ...
```

# Environment Variables
* Enable logging for plugin: PLUG_DEBUG=(all/init/stream/event/mem/off)
* Enable Vulkan validation layers: VULTEN_VALIDATION=(on/off)
* Set device used for benchmarking: VULTEN_BENCH_DEV=(number)
* Disable extended type support: VULTEN_SETTINGS=(DISABLE_INT64)