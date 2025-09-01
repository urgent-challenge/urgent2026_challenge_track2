#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

echo "===== Start preparing [SOMOS] dataset ====="

# download dataset
if [ ! -e ${db}/download.done ]; then
    mkdir -p ${db}
    pushd ${db}
    # wget https://zenodo.org/records/7378801/files/somos.zip
    hf download  --local-dir . --repo-type dataset urgent-challenge/urgent26_track2_sqa somos.zip
    unzip somos.zip
    unzip audios.zip
    rm somos.zip
    rm audios.zip
    popd
    echo "Successfully finished download."
    touch ${db}/download.done
else
    echo "Already exists. Skip download."
fi

mkdir -p ${db}/data data/somos

if [ ! -e data/somos/train/data.jsonl ]; then
    echo "preparing data/somos/train/data.jsonl"
    scripts/data/somos/data_prep.py --generate-listener-id \
        --original-path "${db}/training_files/split1/clean/TRAINSET" --wavdir "${db}/audios" --out "${db}/data/train.csv"
    scripts/data/csv2data.sh "${db}/data/train.csv" "data/somos/train"
fi
if [ ! -e data/somos/dev/data.jsonl ]; then
    echo "preparing data/somos/dev/data.jsonl"
    scripts/data/somos/data_prep.py \
        --original-path "${db}/training_files/split1/clean/VALIDSET" --wavdir "${db}/audios" --out "${db}/data/dev.csv"
    scripts/data/csv2data.sh "${db}/data/dev.csv" "data/somos/dev"
fi
if [ ! -e data/somos/test/data.jsonl ]; then
    echo "preparing data/somos/test/data.jsonl"
    scripts/data/somos/data_prep.py \
        --original-path "${db}/training_files/split1/clean/TESTSET" --wavdir "${db}/audios" --out "${db}/data/test.csv"
    scripts/data/csv2data.sh "${db}/data/test.csv" "data/somos/test"
fi

echo "===== Finished preparing [SOMOS] dataset ====="