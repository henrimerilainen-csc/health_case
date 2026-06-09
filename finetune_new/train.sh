#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=8
#SBATCH --output=./log/finetune_medgemma_mlflow/%j/output.log
#SBATCH --error=./log/finetune_medgemma_mlflow/%j/error.log


module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif         #laifs-lumi-multi-latest.sif # tarkista onko tämä vai lumi-xxxxx lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/hf-cache/hub
mkdir -p $HF_HOME

singularity run $SIF bash -c "python -m venv --system-site-packages ./venv && source ./venv/bin/activate && pip install -U transformers==5.5.4"

# export PYTHONPATH=$PYTHONPATH:./venv/lib/python3.12/site-packages

export HF_TOKEN_PATH=~/.cache/huggingface/token

OUTPUT_DIR=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/health_case/ft_model
mkdir -p $OUTPUT_DIR

MODEL_NAME="google/medgemma-1.5-4b-it"

MLFLOW_MLRUNS_DIR=$OUTPUT_DIR/mlruns
mkdir -p $MLFLOW_MLRUNS_DIR

MLFLOW_EXPERIMENT="${MODEL_NAME##*/}structured-note-finetuned"

# NOTE!!! Use the same json file name you used when submitting the data_mod_vllm job
JSON_FILE=/scratch/${SLURM_JOB_ACCOUNT}/data/structured_notes.json

export TOKENIZERS_PARALLELISM=false

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}" # set port based on SLURM_JOB_ID to avoid conflicts

export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

set -xv

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

srun singularity run $SIF python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --node_rank $SLURM_PROCID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    finetune_new/train.py $* \
    --input-model "$MODEL_NAME" \
    --output-path $OUTPUT_DIR \
    --mlflow_tracking_uri $MLFLOW_MLRUNS_DIR \
    --mlflow_experiment $MLFLOW_EXPERIMENT \
    --json-file $JSON_FILE \
    --model_output_name="${MODEL_NAME##*/}-structured_note_${SLURM_JOB_ID}" \
    --num-workers $SLURM_CPUS_PER_TASK \
    --batch_size=8 \
    --peft \
    # --max-steps=400 \
