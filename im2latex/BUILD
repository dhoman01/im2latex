package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//im2latex/...",
    ],
)

py_binary(
    name = "crop_images",
    srcs = [
        "data/crop_images.py",
    ],
)

py_binary(
    name = "create_tf_records",
    srcs = [
        "data/create_tf_records.py",
    ],
)

sh_binary(
    name = "download_and_preprocess_im2latex",
    srcs = ["data/download_and_preprocess_im2latex.sh"],
    data = [
        ":crop_images",
        ":create_tf_records",
    ],
)

py_library(
    name = "configurations",
    srcs = ["configurations.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//im2latex/ops:image_embeddings",
        "//im2latex/ops:image_processing",
        "//im2latex/ops:inputs",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configurations",
        ":model",
    ],
)

py_library(
    name = "inference_wrapper",
    srcs = ["inference_wrapper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":show_and_tell_model",
        "//im2latex/inference_utils:inference_wrapper_base",
    ],
)

py_binary(
    name = "run_inference",
    srcs = ["run_inference.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":inference_wrapper",
        "//im2latex/inference_utils:caption_generator",
        "//im2latex/inference_utils:vocabulary",
    ],
)
