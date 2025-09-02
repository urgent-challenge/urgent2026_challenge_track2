db=$1

if [ -z "${db}" ]; then
    echo "Usage: $0 <rawdata-dir>"
    exit 1
fi

# following datasets are automatically downloaded and prepared
datasets="tencent somos tmhint-qi chime-7-udase-eval tcd-voip ttsds2 pstn nisqa urgent24-sqa urgent25-sqa"

# following datasets require manual processing after downloading
datasets="${datasets} bvcc bc19" 

hf download --repo-type dataset urgent-challenge/urgent26_track2_sqa data .
mkdir -p ${db}
for dataset in ${datasets}; do
    bash scripts/data/${dataset}/prepare.sh ${db}/${dataset}
done