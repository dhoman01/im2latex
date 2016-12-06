# Location to save the im2latex data.
DATA_DIR="${HOME}/im2latex/data_dir"

# Build the preprocessing script.
bazel build im2latex/download_and_preprocess_im2latex

# Run the preprocessing script.
bazel-bin/im2latex/download_and_preprocess_im2latex "${DATA_DIR}"
