#!/bin/bash
#SBATCH -o log/%j-compound_subset.log
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
    --experiment_name compound_superposition \
    --experiment_type activation_subset \
    --feature_dataset compound_words.pyth.24.-1 \
    --model pythia-1b \
    --batch_size 256 \
    --save_by_neuron \
    --skip_computing_token_summary_df \
    --neuron_subset_file compound_superposition.csv


python get_activations.py \
    --experiment_name compound_superposition \
    --experiment_type activation_subset \
    --feature_dataset pile_test.pyth.512.-1 \
    --model pythia-1b \
    --batch_size 64 \
    --save_by_neuron \
    --skip_computing_token_summary_df \
    --neuron_subset_file compound_superposition.csv


# python get_activations.py \
#     --experiment_name top_compound_words \
#     --experiment_type activation_subset \
#     --feature_dataset pile_test.pyth.512.-1 \
#     --model pythia-160m \
#     --batch_size 128 \
#     --neuron_subset_file top_compound_words.csv


# python get_activations.py \
#     --experiment_name top_compound_words \
#     --experiment_type activation_subset \
#     --feature_dataset pile_test.pyth.512.-1 \
#     --model pythia-2.8b \
#     --batch_size 32 \
#     --neuron_subset_file top_compound_words.csv
