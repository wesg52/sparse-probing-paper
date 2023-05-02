#!/bin/bash
#SBATCH -o log/%j-all_context_features.log
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

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

PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b')

for model in "${PYTHIA_MODELS[@]}"
do
    python -u get_activations.py \
        --experiment_name all_context_features_test \
        --experiment_type activation_probe_dataset \
        --model "$model" \
        --feature_dataset programming_lang_id.pyth.512.-1 \
        --activation_aggregation mean \
        --seed 42

    python -u get_activations.py \
        --experiment_name all_context_features_test \
        --experiment_type activation_probe_dataset \
        --model "$model" \
        --feature_dataset distribution_id.pyth.512.-1 \
        --activation_aggregation mean \
        --seed 42 
    
    python -u get_activations.py \
        --experiment_name all_context_features_test \
        --experiment_type activation_probe_dataset \
        --model "$model" \
        --feature_dataset natural_lang_id.pyth.512.-1 \
        --activation_aggregation mean \
        --seed 42 

    python -u get_activations.py \
        --experiment_name all_context_features_test \
        --experiment_type activation_probe_dataset \
        --model "$model" \
        --feature_dataset natural_lang_id.pyth.512.-1 \
        --seed 42 
done