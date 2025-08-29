db=/mnt/bn/wangwei-nas-lq-03/datasets/urgent26


# following datasets are redistributable, served with huggingface
datasets="tencent somos tmhint-qi chime-7-udase-eval tcd-voip ttsds2"
# ttsds2

# following datasets are not redistributable, we use the original links to download them, you might experience network issues
datasets="${datasets} pstn nisqa"

# following datasets are not redistributable and requires manual configuration,  after downloading, you must manually run the data preparation instructions
datasets="${datasets} bvcc bc19" 

. ./scripts/parse_options.sh

mkdir -p ${db}
for dataset in ${datasets}; do
    bash scripts/data/${dataset}/prepare.sh ${db}/${dataset}
done