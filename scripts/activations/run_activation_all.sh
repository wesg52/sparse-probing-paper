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
    --experiment_name pyth70m_n1_111_stimuli \
    --experiment_type all_activations \
    --feature_dataset neuron_stimulus.pyth.32.-1 \
    --batch_size 64 \
    --model pythia-70m \
    --layers 1 \
    --positions 31
