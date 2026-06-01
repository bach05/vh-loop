#!/bin/bash
#SBATCH -o logs/outputs/%x_%j.out
#SBATCH -e logs/errors/%x_%j.err
#SBATCH -t 96:00:00
#SBATCH -n 2
#SBATCH -c 2
#SBATCH -p allgroups
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=32G
#SBATCH --mail-user=bacchinalb@dei.unipd.it
#SBATCH --mail-type=ALL

set -euo pipefail

ENTRYPOINT_YAML="${1:?Missing entrypoint yaml}"
TRAINER_CONFIG_YAML="${2:?Missing trainer config yaml}"
MODEL_CONFIG_YAML="${3:?Missing model config yaml}"
RUN_ID="${4:?Missing run_id}"

WORKDIR_HOST="/nfsd/iaslab4/Users/bacchin"
REPO_HOST="${WORKDIR_HOST}/vhl_ws/vh-loop"
SIF_HOST="${REPO_HOST}/containers/singularity/vhl.sif"
CACHE_HOST="${WORKDIR_HOST}/model_cache"


# real cluster-visible data root
DATA_BIND_SRC="/nfsd/iaslab4/Users/bacchin/datasets"

# Writable Pixi folders on the host. Avoid to rebuild sif every time Pixi needs to update its envs or cache, which happens at every run otherwise.
# This fixes the read-only error caused by Pixi trying to write under /opt/pixi/envs inside the SIF.
PIXI_ENV_HOST="${CACHE_HOST}/pixi/envs"
PIXI_CACHE_HOST="${CACHE_HOST}/pixi/cache"

#mkdir -p "${PIXI_ENV_HOST}"
mkdir -p "${PIXI_CACHE_HOST}"
mkdir -p "${CACHE_HOST}/huggingface"
mkdir -p "${CACHE_HOST}/xdg"
mkdir -p "${REPO_HOST}/logs/train_sample"
mkdir -p "${REPO_HOST}/logs/outputs"
mkdir -p "${REPO_HOST}/logs/errors"

HOST_LOG="${REPO_HOST}/logs/train_sample/${RUN_ID}.log"

# Environment variables passed inside Apptainer/Singularity.
# Both APPTAINERENV_* and SINGULARITYENV_* are set for compatibility.
export APPTAINERENV_HF_HOME=/models/huggingface
export APPTAINERENV_HF_HUB_CACHE=/models/huggingface/hub
export APPTAINERENV_TRANSFORMERS_CACHE=/models/huggingface/hub
export APPTAINERENV_XDG_CACHE_HOME=/models/xdg

export APPTAINERENV_PIXI_CACHE_DIR=/models/pixi/cache
export APPTAINERENV_PIXI_FROZEN=true

# Optional: keep repodata cache local to the compute node to avoid Pixi/rattler NFS warnings.
export APPTAINERENV_PIXI_CACHE_REPODATA_DIR="/tmp/pixi-repodata-${SLURM_JOB_ID:-manual}"

export SINGULARITYENV_HF_HOME="${APPTAINERENV_HF_HOME}"
export SINGULARITYENV_HF_HUB_CACHE="${APPTAINERENV_HF_HUB_CACHE}"
export SINGULARITYENV_TRANSFORMERS_CACHE="${APPTAINERENV_TRANSFORMERS_CACHE}"
export SINGULARITYENV_XDG_CACHE_HOME="${APPTAINERENV_XDG_CACHE_HOME}"

export SINGULARITYENV_PIXI_CACHE_DIR="${APPTAINERENV_PIXI_CACHE_DIR}"
export SINGULARITYENV_PIXI_FROZEN="${APPTAINERENV_PIXI_FROZEN}"
export SINGULARITYENV_PIXI_CACHE_REPODATA_DIR="${APPTAINERENV_PIXI_CACHE_REPODATA_DIR}"

echo "=================================================="
echo "RUN_ID        : ${RUN_ID}"
echo "ENTRYPOINT    : ${ENTRYPOINT_YAML}"
echo "MODEL         : ${MODEL_CONFIG_YAML}"
echo "TRAINER       : ${TRAINER_CONFIG_YAML}"
echo "REPO_HOST     : ${REPO_HOST}"
echo "SIF_HOST      : ${SIF_HOST}"
echo "DATA_BIND_SRC : ${DATA_BIND_SRC}"
echo "CACHE_HOST    : ${CACHE_HOST}"
#echo "PIXI_ENV_HOST : ${PIXI_ENV_HOST}"
echo "PIXI_CACHE    : ${PIXI_CACHE_HOST}"
echo "HOST_LOG      : ${HOST_LOG}"
echo "SLURM_JOB_ID  : ${SLURM_JOB_ID:-manual}"
echo "=================================================="

cd "${REPO_HOST}"

singularity exec \
  --cleanenv \
  --nv \
  --bind "${REPO_HOST}:/workspace/vh-loop" \
  --bind "${DATA_BIND_SRC}:/data" \
  --bind "${CACHE_HOST}:/models" \
  "${SIF_HOST}" \
  bash -lc '
    export PYTHONPATH=/workspace/vh-loop:/mnt/vh-loop:${PYTHONPATH}
    cd /workspace/vh-loop

    pixi run --manifest-path /mnt/vh-loop \
      python tests/train_sample.py \
        --config-name "'"${ENTRYPOINT_YAML}"'" \
        trainer="'"${TRAINER_CONFIG_YAML}"'" \
        model="'"${MODEL_CONFIG_YAML}"'"
  ' 2>&1 | tee "${HOST_LOG}"


# Examle command to submit the job: 
# sbatch hpc/slurm/train_sample.slurm train_entrypoint.yaml qwen_sft_trainer_CLUSTER qwen3_5_CLUSTER qwen_4B