#!/bin/bash
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -o log/ewt_fast_probe.log-%j-%a
#SBATCH -a 1-16


# set environment variables
export PATH=$SPARSE_PROBING_ROOT:$PATH

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/home/gridsan/groups/maia_mechint/sparse_probing/results
export FEATURE_DATASET_DIR=/home/gridsan/groups/maia_mechint/sparse_probing/feature_datasets
export TRANSFORMERS_CACHE=/home/gridsan/groups/maia_mechint/models
export HF_DATASETS_CACHE=/home/gridsan/groups/maia_mechint/sparse_probing/hf_home
export HF_HOME=/home/gridsan/groups/maia_mechint/sparse_probing/hf_home

sleep 0.1  # wait for paths to update

# activate environment and load modules
source $SPARSE_PROBING_ROOT/sparprob/bin/activate
source /etc/profile
module load gurobi/gurobi-1000


PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b')

for model in "${PYTHIA_MODELS[@]}"
do
    python -u probing_experiment.py \
        --experiment_name ewt_test \
        --experiment_type fast_heuristic_sparsity_sweep enumerate_monosemantic \
        --model "$model" \
        --feature_dataset ewt.pyth.512.-1 \
        --save_features_together
done

for model in "${PYTHIA_MODELS[@]}"
do
    python -u probing_experiment.py \
        --experiment_name ewt_test \
        --experiment_type fast_heuristic_sparsity_sweep enumerate_monosemantic \
        --model "$model" \
        --feature_dataset ewt.pyth.512.-1 \
        --save_features_together \
        --probe_next_token_feature
done
