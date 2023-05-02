#!/bin/bash
#SBATCH -o log/%j-all_context_subset.log
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
    --experiment_name context_monosemantic \
    --experiment_type activation_subset \
    --feature_dataset natural_lang_id.pyth.512.-1 \
    --model pythia-70m \
    --batch_size 256 \
    --neuron_subset_file monosemantic_language_neurons.csv


python get_activations.py \
    --experiment_name context_monosemantic \
    --experiment_type activation_subset \
    --feature_dataset programming_lang_id.pyth.512.-1 \
    --model pythia-1b \
    --batch_size 128 \
    --neuron_subset_file monosemantic_code_neurons.csv


python get_activations.py \
    --experiment_name context_monosemantic \
    --experiment_type activation_subset \
    --feature_dataset distribution_id.pyth.512.-1 \
    --model pythia-6.9b \
    --batch_size 32 \
    --neuron_subset_file monosemantic_distribution_neurons.csv


python get_activations.py \
    --experiment_name context_monosemantic \
    --experiment_type activation_subset \
    --feature_dataset pile_test.pyth.512.-1 \
    --model pythia-70m \
    --batch_size 256 \
    --neuron_subset_file monosemantic_language_neurons.csv


python get_activations.py \
    --experiment_name context_monosemantic \
    --experiment_type activation_subset \
    --feature_dataset pile_test.pyth.512.-1 \
    --model pythia-1b \
    --batch_size 128 \
    --neuron_subset_file monosemantic_code_neurons.csv
