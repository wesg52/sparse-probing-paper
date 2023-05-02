#!/bin/bash
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -o log/all_context_features.log-%j-%a
#SBATCH -a 1-12


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

PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b' 'pythia-12b')
PYTHIA_MODELS_MEDIUM=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b')
PYTHIA_MODELS_LARGE=('pythia-2.8b' 'pythia-6.9b')
PYTHIA_MODELS_XL=('pythia-12b')

CONTEXT_FEATURE_DATASETS=('programming_lang_id.pyth.512.-1' 'distribution_id.pyth.512.-1' 'natural_lang_id.pyth.512.-1')

for model in "${PYTHIA_MODELS_MEDIUM[@]}"
do
    python -u probing_experiment.py \
        --experiment_name all_context_features \
        --experiment_type enumerate_monosemantic fast_heuristic_sparsity_sweep \
        --model "$model" \
        --feature_dataset programming_lang_id.pyth.512.-1 \
        --activation_aggregation mean \
        --normalize_activations \
        --seed 42 \
        --save_features_together

    python -u probing_experiment.py \
        --experiment_name all_context_features \
        --experiment_type enumerate_monosemantic fast_heuristic_sparsity_sweep \
        --model "$model" \
        --feature_dataset distribution_id.pyth.512.-1 \
        --activation_aggregation mean \
        --normalize_activations \
        --seed 42 \
        --save_features_together

    python -u probing_experiment.py \
        --experiment_name all_context_features \
        --experiment_type enumerate_monosemantic fast_heuristic_sparsity_sweep \
        --model "$model" \
        --feature_dataset natural_lang_id.pyth.512.-1 \
        --activation_aggregation mean \
        --normalize_activations \
        --seed 42 \
        --save_features_together
done