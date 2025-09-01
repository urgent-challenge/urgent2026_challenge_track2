#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

echo "===== Start preparing [BVCC] dataset ====="

# download dataset
cwd=`pwd`
if [ ! -e ${db}/download.done ]; then
    mkdir -p ${db}
    pushd ${db}
    wget https://zenodo.org/records/6572573/files/main.tar.gz
    tar zxvf main.tar.gz
    rm main.tar.gz
    popd
    echo "Successfully finished download. Please follow the instructions."
    touch ${db}/download.done
else
    echo "Already exists. Skip download."
fi


num_files=$(find ${db}/main/DATA/wav -type f -name "*.wav" | wc -l)

if [ $num_files -eq 0 ]; then
    echo "No wav files found in ${db}/main/DATA/wav."
    echo "The BVCC dataset requires manual download due to licensing restrictions."
    echo "Run 01_get.py, 02_gather.py, and 03_preprocess.py scripts in ${db}/main prepare the dataset."
    exit 1
fi

if [ $num_files -ne 7106 ]; then
    echo "Error: Expected 7106 wav files in ${db}/main/DATA/wav, but found ${num_files}."
    exit 1
fi

mkdir -p ${db}/data data/bvcc
if [ ! -e data/bvcc/train/data.jsonl ]; then
    scripts/data/bvcc/data_prep.py --generate-listener-id \
        --original-path "${db}/main/DATA/sets/TRAINSET" --wavdir "${db}/main/DATA/wav" --out "${db}/data/train.csv"
    scripts/data/csv2data.sh "${db}/data/train.csv" "data/bvcc/train"
fi
if [ ! -e data/bvcc/dev/data.jsonl ]; then
    scripts/data/bvcc/data_prep.py --generate-listener-id \
        --original-path "${db}/main/DATA/sets/DEVSET" --wavdir "${db}/main/DATA/wav" --out "${db}/data/dev.csv"
    scripts/data/csv2data.sh "${db}/data/dev.csv" "data/bvcc/dev"
fi
if [ ! -e data/bvcc/test/data.jsonl ]; then
    scripts/data/bvcc/data_prep.py --generate-listener-id \
        --original-path "${db}/main/DATA/sets/TESTSET" --wavdir "${db}/main/DATA/wav" --out "${db}/data/test.csv"
    scripts/data/csv2data.sh "${db}/data/test.csv" "data/bvcc/test"
fi

echo "===== Finished preparing [BVCC] dataset ====="