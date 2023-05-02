#!/bin/bash
#SBATCH -o log/activation_all.log-%j
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

# setup env
export PATH=$SPARSE_PROBING_ROOT:$PATH

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/home/gridsan/groups/maia_mechint/sparse_probing/results
export FEATURE_DATASET_DIR=/home/gridsan/groups/maia_mechint/sparse_probing/feature_datasets
export TRANSFORMERS_CACHE=/home/gridsan/groups/maia_mechint/models
export HF_DATASETS_CACHE=/home/gridsan/groups/maia_mechint/sparse_probing/hf_home
export HF_HOME=/home/gridsan/groups/maia_mechint/sparse_probing/hf_home

sleep 0.1 # wait for paths to update

source $SPARSE_PROBING_ROOT/sparprob/bin/activate
source /etc/profile

python get_activations.py \
    --experiment_name ewt_full \
    --experiment_type all_activations \
    --feature_dataset preprocessed_ewt_512.hf \
    --batch_size 64 \
    --model pythia-70m \
    --flatten_and_ignore_padding


python get_activations.py \
    --experiment_name ewt_full \
    --experiment_type all_activations \
    --feature_dataset preprocessed_ewt_512.hf \
    --batch_size 64 \
    --model pythia-1b \
    --flatten_and_ignore_padding
