#!/usr/bin/env bash
set -e

db=$1

echo "===== Start preparing [URGENT25-SQA] dataset ====="

# download dataset
cwd=`pwd`
if [ ! -e ${db}/download.done ]; then
    mkdir -p ${db}
    pushd ${db}
    # hf download --local-dir . --repo-type dataset urgent-challenge/URGENT25-SQA
    # rm ood.tar.gz
    popd
    echo "Successfully finished download."
    touch ${db}/download.done
else
    echo "Already exists. Skip download."
fi


mkdir -p ${db}/data data/urgent25-sqa

if [ ! -f data/urgent25-sqa/train/data.jsonl ]; then
    scripts/data/urgent25-sqa/data_prep.py --data "${db}" --split "train" --out "${db}/data/train.csv"
    scripts/data/csv2data.sh "${db}/data/train.csv" "data/urgent25-sqa/train"
fi

echo "===== Finished preparing [URGENT25-SQA] dataset ====="