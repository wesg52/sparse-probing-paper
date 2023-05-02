#!/bin/bash
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o log/%j-sequence_ablation.log


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


python run_ablation.py \
    --feature_dataset natural_lang_id.pyth.512.-1 \
    --model pythia-70m \
    --batch_size 64 \
    --neuron_subset_file monosemantic_language_neurons.csv


python run_ablation.py \
    --feature_dataset programming_lang_id.pyth.512.-1 \
    --model pythia-1b \
    --batch_size 32 \
    --neuron_subset_file monosemantic_code_neurons.csv


python run_ablation.py \
    --feature_dataset distribution_id.pyth.512.-1 \
    --model pythia-6.9b \
    --batch_size 8 \
    --neuron_subset_file monosemantic_distribution_neurons.csv


python run_ablation.py \
    --feature_dataset pile_test.pyth.512.-1 \
    --model pythia-70m \
    --batch_size 64 \
    --neuron_subset_file monosemantic_language_neurons.csv


python run_ablation.py \
    --feature_dataset pile_test.pyth.512.-1 \
    --model pythia-1b \
    --batch_size 32 \
    --neuron_subset 6,3108 9,7926 10,3855 9,1693