#!/bin/bash
#SBATCH -o log/%j-save_top_mono_neurons.log
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

PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b')

for model in "${PYTHIA_MODELS[@]}"
do
    python get_activations.py \
        --experiment_name top_mono_neurons \
        --experiment_type activation_subset \
        --feature_dataset programming_lang_id.pyth.512.-1 \
        --model "$model" \
        --neuron_subset_file top_mono_neurons.csv \
        --auto_restrict_neuron_subset_file
    
    python get_activations.py \
        --experiment_name top_mono_neurons \
        --experiment_type activation_subset \
        --feature_dataset distribution_id.pyth.512.-1 \
        --model "$model" \
        --neuron_subset_file top_mono_neurons.csv \
        --auto_restrict_neuron_subset_file

#     python get_activations.py \
#         --experiment_name top_mono_neurons \
#         --experiment_type activation_subset \
#         --feature_dataset natural_lang_id.pyth.512.-1 \
#         --model "$model" \
#         --neuron_subset_file top_mono_neurons.csv \
#         --auto_restrict_neuron_subset_file \
#         --skip_computing_token_summary_df

    python get_activations.py \
        --experiment_name top_mono_neurons \
        --experiment_type activation_subset \
        --feature_dataset compound_words.pyth.24.-1 \
        --model "$model" \
        --neuron_subset_file top_mono_neurons.csv \
        --auto_restrict_neuron_subset_file \
        --skip_computing_token_summary_df

    python get_activations.py \
        --experiment_name top_mono_neurons \
        --experiment_type activation_subset \
        --feature_dataset text_features.pyth.256.10000 \
        --model "$model" \
        --neuron_subset_file top_mono_neurons.csv \

    python get_activations.py \
        --experiment_name top_mono_neurons \
        --experiment_type activation_subset \
        --feature_dataset ewt.pyth.512.-1 \
        --model "$model" \
        --neuron_subset_file top_mono_neurons.csv \
        --auto_restrict_neuron_subset_file \
        --skip_computing_token_summary_df

    python get_activations.py \
        --experiment_name top_mono_neurons \
        --experiment_type activation_subset \
        --feature_dataset latex.pyth.1024.-1 \
        --model "$model" \
        --batch_size 8 \
        --neuron_subset_file top_mono_neurons.csv \
        --auto_restrict_neuron_subset_file
done