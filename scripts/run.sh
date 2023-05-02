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

# Text features sweep across all models
for model in "${PYTHIA_MODELS[@]}"
do
    python probing_experiment.py \
        --experiment_name code_lang_max_test \
        --experiment_type heuristic_sparsity_sweep\
        --model "$model" \
        --feature_dataset programming_lang_id.pyth.512.-1 \
        --activation_aggregation max
done


for model in "${PYTHIA_MODELS[@]}"
do
    python probing_experiment.py \
        --experiment_name nat_lang_test \
        --experiment_type heuristic_sparsity_sweep\
        --model "$model" \
        --feature_dataset natural_lang_id.pyth.512.-1
done



# python probing_experiment.py \
#     --experiment_name language_id_test \
#     --experiment_type heuristic_sparsity_sweep\
#     --model pythia-19m \
#     --feature_dataset github_lang_id.test.pyth.512.-1.True
#     --average_seq_activations


# python probing_experiment.py \
#  --experiment_name heuristic_feature_selection_test \
#  --experiment_type test_heuristic_filtering \
#  --feature_datasets true_binary_token_supervised_feature_datasets \
#  --n_seqs 1000 \
#  --osp_upto_k 12 \
#  --gurobi_timeout 600
