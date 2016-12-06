#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Script to download and preprocess the im2latex data set.
#
# The outputs of this script are sharded TFRecord files containing serialized
# SequenceExample protocol buffers. See build_mscoco_data.py for details of how
# the SequenceExample protocol buffers are constructed.
#
# usage:
#  ./download_and_preprocess_im2latex.sh
set -e

if [ -z "$1" ]; then
  echo "usage download_and_preproces_im2latex.sh [data dir]"
  exit
fi

UNZIP="tar -xf"

# Create the output directories.
OUTPUT_DIR="${1%/}"
SCRATCH_DIR="${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"
CURRENT_DIR=$(pwd)
WORK_DIR="$0.runfiles/im2txt/im2txt"

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  ${UNZIP} ${FILENAME}
}

function download(){
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
}

cd ${SCRATCH_DIR}

BASE_URL="https://zenodo.org/record/56198/files"

# Download the images.
TRAIN_IMAGE_FILE="formula_images.tar.gz"
download_and_unzip ${BASE_URL} ${TRAIN_IMAGE_FILE}

# Download the formulas.
FORMULAS_FILE="im2latex_formulas.lst"
download ${BASE_URL} ${FORMULAS_FILE}


# Download train, val, and test LST
TRAIN_FILE="im2latex_train.lst"
download ${BASE_URL} ${TRAIN_FILE}

VAL_FILE="im2latex_validate.lst"
download ${BASE_URL} ${VAL_FILE}

TEST_FILE="im2latex_test.lst"
download ${BASE_URL} ${TEST_FILE}

# Build TFRecords of the image data.
cd "${CURRENT_DIR}"
BUILD_SCRIPT="${WORK_DIR}/create_tf_records"
"${BUILD_SCRIPT}" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/word_counts.txt" \
