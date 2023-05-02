#!/bin/bash
#SBATCH -o log/activation_metrics.log-%j
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

PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b')

for model in "${PYTHIA_MODELS[@]}"
do
python get_activations.py \
    --experiment_name full_range \
    --experiment_type full_activation_histogram \
    --feature_dataset pile_test.pyth.512.-1 \
    --n_bin 1000 --hist_max 100 --hist_min -100 \
    --model "$model"
done