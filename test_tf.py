import tensorflow as tf

print(tf.config.list_physical_devices())

print(tf.__version__)

#clear && TF_PLUGGABLE_DEVICE_LIBRARY_PATH=target/debug/libvulten_rs.so VULTEN_VALIDATION=off PLUG_DEBUG=off python3 test_tf.py
