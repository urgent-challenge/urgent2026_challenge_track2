#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

echo "===== Start preparing [TMHINT-QI] dataset ====="
# download dataset
cwd=`pwd`
if [ ! -e ${db}/download.done ]; then
    mkdir -p ${db}
    pushd ${db}
    # gdown 1TMDiz6dnS76hxyeAcCQxeSqqEOH4UDN0
    hf download  --local-dir . --repo-type dataset urgent-challenge/urgent26_track2_sqa TMHINTQI.zip
    unzip TMHINTQI.zip
    rm -rf __MACOSX/
    mv TMHINTQI/* .
    rm -rf TMHINTQI
    # rm TMHINTQI.zip
    popd
    echo "Successfully finished download."
    touch ${db}/download.done
else
    echo "Already exists. Skip download."
fi


mkdir -p ${db}/data data/tmhint-qi

if [ ! -e data/tmhint-qi/train/data.jsonl ]; then
    scripts/data/tmhint-qi/data_prep.py \
        --original-path "${db}/raw_data.csv" --wavdir "${db}/train" --setname "train" --out "${db}/data/train.csv" --seed 1337 --dev_ratio 0.0
    scripts/data/csv2data.sh "${db}/data/train.csv" "data/tmhint-qi/train"
fi
if [ ! -e data/tmhint-qi/test/data.jsonl ]; then
    scripts/data/tmhint-qi/data_prep.py \
        --original-path "${db}/raw_data.csv" --wavdir "${db}/test" --setname "test" --out "${db}/data/test.csv" --seed 1337
    scripts/data/csv2data.sh "${db}/data/test.csv" "data/tmhint-qi/test"
fi

echo "===== Finished preparing [TMHINTQI] dataset ====="