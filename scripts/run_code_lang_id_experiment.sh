#!/bin/bash
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -o log/runtest.log-%j-%a
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
module load gurobi/gurobi-951


PYTHIA_MODELS=('pythia-19m' 'pythia-125m' 'pythia-350m' 'pythia-800m' 'pythia-1.3b')

python probing_experiment.py \
    --experiment_name code_lang_max_test \
    --experiment_type heuristic_sparsity_sweep\
    --model pythia-800m \
    --feature_dataset programming_lang_id.pyth.512.-1 \
    --activation_aggregation max

python probing_experiment.py \
    --experiment_name code_lang_test \
    --experiment_type heuristic_sparsity_sweep\
    --model pythia-800m \
    --feature_dataset programming_lang_id.pyth.512.-1 \
    --probe_location hook_resid_post

python probing_experiment.py \
    --experiment_name code_lang_test \
    --experiment_type heuristic_sparsity_sweep\
    --model pythia-800m \
    --feature_dataset programming_lang_id.pyth.512.-1


