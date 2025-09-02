db=$1

if [ -z "${db}" ]; then
    echo "Usage: $0 <rawdata-dir>"
    exit 1
fi

# following datasets are automatically downloaded and prepared
datasets="tencent somos tmhint-qi chime-7-udase-eval tcd-voip ttsds2 pstn nisqa urgent2024-sqa urgent2025-sqa"

# following datasets require manual processing after downloading
datasets="${datasets} bvcc bc19" 

if [ ! -d "./data" ]; then
    hf download --repo-type dataset urgent-challenge/urgent26_track2_sqa data.zip --local-dir .
    unzip data.zip
    rm -f data.zip
fi
mkdir -p ${db}
for dset in ${datasets}; do
    bash scripts/data/${dset}/prepare.sh ${db}/${dset}
done