#!/usr/bin/env bash
set -e

db=$1

echo "===== Start preparing [CHiME-7-UDASE-EVAL] dataset ====="

# download dataset
cwd=`pwd`
if [ ! -e ${db}/chime-7-udase-eval.done ]; then
    mkdir -p ${db}
    pushd ${db}
    # wget -c https://zenodo.org/records/10418311/files/CHiME-7-UDASE-evaluation-data.zip
    hf download  --local-dir . --repo-type dataset urgent-challenge/urgent26_track2_sqa CHiME-7-UDASE-evaluation-data.zip
    unzip CHiME-7-UDASE-evaluation-data.zip
    # rm CHiME-7-UDASE-evaluation-data.zip
    popd
    echo "Successfully finished download."
    touch ${db}/chime-7-udase-eval.done
else
    echo "Already exists. Skip download."
fi

mkdir -p ${db}/data data/chime-7-udase-eval

if [ ! -e data/chime-7-udase-eval/test/wav.scp ]; then
    scripts/data/chime-7-udase-eval/data_prep.py \
        --original-path "${db}/listening_test/MOS_results_listening_test.csv" --wavdir "${db}/listening_test/data" --out "${db}/data/chime-7-udase-eval_test.csv"
    scripts/data/csv2scps.py "${db}/data/chime-7-udase-eval_test.csv" "data/chime-7-udase-eval/test"
fi

echo "===== Finished preparing [CHiME-7-UDASE-EVAL] dataset ====="