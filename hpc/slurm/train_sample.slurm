#!/bin/bash
#SBATCH -o logs/outputs/%x_%j.out
#SBATCH -e logs/errors/%x_%j.err
#SBATCH -t 24:00:00
#SBATCH -n 2
#SBATCH -c 2
#SBATCH -p allgroups
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=8G
#SBATCH --mail-user=bacchinalb@dei.unipd.it
#SBATCH --mail-type=ALL

set -euo pipefail

ENTRYPOINT_YAML="${1:?Missing entrypoint yaml}"
TRAINER_CONFIG_TRAINER_CONFIG_YAMLYAML="${2:?Missing trainer config yaml}"
MODEL_CONFIG_YAML="${3:?Missing model config yaml}"
RUN_ID="${4:?Missing run_id}"

WORKDIR_HOST="/nfsd/iaslab4/Users/bacchin"
REPO_HOST="${WORKDIR_HOST}/vhl_ws/vh-loop"
SIF_HOST="${REPO_HOST}/containers/vhl.sif"
CACHE_HOST="${WORKDIR_HOST}/model_cache"


# real cluster-visible data root
DATA_BIND_SRC="/nfsd/iaslab4/Users/bacchin/datasets"

mkdir -p "${REPO_HOST}/logs/train_sample"

HOST_LOG="${REPO_HOST}/logs/train_sample/${RUN_ID}.log"

# set paths in the container environment
export APPTAINERENV_HF_HOME=/models/huggingface
export APPTAINERENV_HF_HUB_CACHE=/models/huggingface/hub
export APPTAINERENV_TRANSFORMERS_CACHE=/models/huggingface/hub
export APPTAINERENV_XDG_CACHE_HOME=/models/xdg

echo "=================================================="
echo "RUN_ID        : ${RUN_ID}"
echo "ENTRYPOINT    : ${ENTRYPOINT_YAML}"
echo "MODEL         : ${MODEL_CONFIG_YAML}"
echo "TRAINER       : ${TRAINER_CONFIG_YAML}"
echo "HOST_LOG      : ${HOST_LOG}"
echo "=================================================="

cd "${REPO_HOST}"

singularity exec \
  --cleanenv \
  --nv \
  --bind "${REPO_HOST}:/mnt/vh-loop" \
  --bind "${DATA_BIND_SRC}:/data" \
  --bind "${CACHE_HOST}:/models" \
  "${SIF_HOST}" \
  pixi run --manifest-path /mnt/vh-loop \
  python /mnt/vh-loop/tests/train_sample.py \
    --config-name "${ENTRYPOINT_YAML}" \
    "trainer=${TRAINER_CONFIG_YAML}" \
    "model=${MODEL_CONFIG_YAML}" \
  2>&1 | tee "${HOST_LOG}"