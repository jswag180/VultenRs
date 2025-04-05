tf_path=$(python -c "import site; print(site.getsitepackages()[0])")/tensorflow

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$tf_path VULTEN_VALIDATION=on VULTEN_TEST_DEV=$1 cargo test -- --test-threads=1 $2 $3
