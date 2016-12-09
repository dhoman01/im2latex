# Directory containing preprocessed im2latex data.
DATA_DIR="${HOME}/im2latex/data_dir"

# Directory to save the model.
MODEL_DIR="${HOME}/im2latex/model"

# Build the model.
bazel build -c opt im2latex/...

# Run the training script.
until bazel-bin/im2latex/train \
  --input_file_pattern="${DATA_DIR}/train-?????-of-00256.tfrecords" \
  --train_dir="${MODEL_DIR}/train" \
  --number_of_steps=500000 >&2; do
    echo "Restarting due to failure..."
    sleep 1
done
