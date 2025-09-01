#!/usr/bin/env bash
set -e

db=$1

echo "===== Start preparing [URGENT24-SQA] dataset ====="

# download dataset
cwd=`pwd`
if [ ! -e ${db}/download.done ]; then
    mkdir -p ${db}
    pushd ${db}
    hf download --local-dir . --repo-type dataset urgent-challenge/URGENT24-SQA
    # rm ood.tar.gz
    popd
    echo "Successfully finished download."
    touch ${db}/download.done
else
    echo "Already exists. Skip download."
fi


mkdir -p ${db}/data data/urgent24-sqa

if [ ! -f data/urgent24-sqa/train/data.jsonl ]; then
    scripts/data/urgent24-sqa/data_prep.py --data "${db}" --split "train" --out "${db}/data/train.csv"
    scripts/data/csv2data.sh "${db}/data/train.csv" "data/urgent24-sqa/train"
fi

if [ ! -f data/urgent24-sqa/test/data.jsonl ]; then
    scripts/data/urgent24-sqa/data_prep.py --data "${db}" --split "test" --out  "${db}/data/test.csv"
    scripts/data/csv2data.sh "${db}/data/test.csv" "data/urgent24-sqa/test"
fi

echo "===== Finished preparing [URGENT24-SQA] dataset ====="