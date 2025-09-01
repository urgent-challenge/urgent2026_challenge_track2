#!/usr/bin/env bash
set -e

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

db=$1

echo "===== Start preparing [NISQA] dataset ====="

# download dataset
cwd=`pwd`
if [ ! -e ${db}/download.done ]; then
    mkdir -p ${db}
    pushd ${db}
    # wget -c https://depositonce.tu-berlin.de/bitstream/11303/13012.5/9/NISQA_Corpus.zip
    hf download  --local-dir . --repo-type dataset urgent-challenge/urgent26_track2_sqa NISQA_Corpus.zip
    unzip NISQA_Corpus.zip
    rm -f NISQA_Corpus.zip
    mv NISQA_Corpus/* .
    rm -rf NISQA_Corpus/
    popd
    echo "Successfully finished download."
    touch ${db}/download.done
else
    echo "Already exists. Skip download."
fi

mkdir -p ${db}/data data/nisqa

if [ ! -e data/nisqa/train/data.jsonl ]; then
    scripts/data/nisqa/data_prep.py \
        --original-path "${db}/NISQA_TRAIN_SIM/NISQA_TRAIN_SIM_file.csv" --wavdir "${db}/NISQA_TRAIN_SIM/deg" --out "${db}/data/train_sim.csv"
    scripts/data/nisqa/data_prep.py \
        --original-path "${db}/NISQA_TRAIN_LIVE/NISQA_TRAIN_LIVE_file.csv" --wavdir "${db}/NISQA_TRAIN_LIVE/deg" --out "${db}/data/train_live.csv"
    scripts/data/nisqa/combine_datasets.py --original-paths "${db}/data/train_sim.csv" "${db}/data/train_live.csv" --out "${db}/data/train.csv"
    scripts/data/csv2data.sh "${db}/data/train.csv" "data/nisqa/train"
fi

if [ ! -e data/nisqa/dev/data.jsonl ]; then
    scripts/data/nisqa/data_prep.py \
        --original-path "${db}/NISQA_VAL_SIM/NISQA_VAL_SIM_file.csv" --wavdir "${db}/NISQA_VAL_SIM/deg" --out "${db}/data/dev_sim.csv"
    scripts/data/nisqa/data_prep.py \
        --original-path "${db}/NISQA_VAL_LIVE/NISQA_VAL_LIVE_file.csv" --wavdir "${db}/NISQA_VAL_LIVE/deg" --out "${db}/data/dev_live.csv"
    scripts/data/nisqa/combine_datasets.py --original-paths "${db}/data/dev_sim.csv" "${db}/data/dev_live.csv" --out "${db}/data/dev.csv"
    scripts/data/csv2data.sh "${db}/data/dev.csv" "data/nisqa/dev"
fi

for test_set in LIVETALK FOR P501; do
    if [ ! -e data/nisqa/${test_set}/data.jsonl ]; then
        scripts/data/nisqa/data_prep.py \
            --original-path "${db}/NISQA_TEST_${test_set}/NISQA_TEST_${test_set}_file.csv" --wavdir "${db}/NISQA_TEST_${test_set}/deg" --out "${db}/data/${test_set}.csv"
        scripts/data/csv2data.sh "${db}/data/${test_set}.csv" "data/nisqa/${test_set}"
    fi
done


echo "===== Finished preparing [NISQA] dataset ====="