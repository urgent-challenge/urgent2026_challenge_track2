#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

echo "===== Start preparing [BC19] dataset ====="

# download dataset
cwd=`pwd`
if [ ! -e ${db}/bc19.done ]; then
    mkdir -p ${db}
    pushd ${db}
    wget https://zenodo.org/records/6572573/files/ood.tar.gz
    tar zxvf ood.tar.gz
    # rm ood.tar.gz
    popd
    echo "Successfully finished download. Please follow the inside ${db}"
    touch ${db}/bc19.done
else
    echo "Already exists. Skip download."
fi


num_files=$(find ${db}/ood/DATA/wav -type f -name "*.wav" | wc -l)
if [ $num_files -eq 0 ]; then
    echo "No wav files found in ${db}/ood/DATA/wav."
    echo "The BC19 dataset requires manual download due to licensing restrictions."
    echo "Run 01_get.py, 02_gather.py, and 03_preprocess.py scripts in ${db}/ood to prepare the dataset."
    exit 1
fi

if [ $num_files -ne 1352 ]; then
    echo "Error: Expected 1352 wav files in ${db}/ood/DATA/wav, but found ${num_files}."
    exit 1
fi

mkdir -p ${db}/data data/bc19
if [ ! -e data/bc19/train.jsonl ]; then
    scripts/data/bc19/data_prep.py \
        --original-path "${db}/ood/DATA/sets/TRAINSET" --wavdir "${db}/ood/DATA/wav" --out "${db}/data/bc19_train.csv"
    scripts/data/csv2jsonl.py "${db}/data/bc19_train.csv" "data/bc19/train.jsonl"
fi
if [ ! -e data/bc19/dev.jsonl ]; then
    scripts/data/bc19/data_prep.py \
        --original-path "${db}/ood/DATA/sets/DEVSET" --wavdir "${db}/ood/DATA/wav" --out "${db}/data/bc19_dev.csv"
    scripts/data/csv2jsonl.py "${db}/data/bc19_dev.csv" "data/bc19/dev.jsonl"
fi
if [ ! -e data/bc19/test.jsonl ]; then
    scripts/data/bc19/data_prep.py \
        --original-path "${db}/ood/DATA/sets/TESTSET" --wavdir "${db}/ood/DATA/wav" --out "${db}/data/bc19_test.csv"
    scripts/data/csv2jsonl.py "${db}/data/bc19_test.csv" "data/bc19/test.jsonl"
fi

echo "===== Finished preparing [BC19] dataset ====="