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
    name = "create_tf_records",
    srcs = [
        "data/create_tf_records.py",
    ],
)

sh_binary(
    name = "download_and_preprocess_im2latex",
    srcs = ["data/download_and_preprocess_im2latex.sh"],
    data = [
        ":create_tf_records",
    ],
)
