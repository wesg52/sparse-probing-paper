#!/bin/bash
#SBATCH -o log/%j-top_fact_neurons.log
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

PYTHIA_MODELS=('pythia-6.9b')

for model in "${PYTHIA_MODELS[@]}"
do
    # python -u get_activations.py \
    #     --experiment_name top_fact_neurons \
    #     --experiment_type activation_subset \
    #     --feature_dataset wikidata_sorted_occupation_athlete.pyth.128.5000 \
    #     --model "$model" \
    #     --neuron_subset_file top_fact_neurons.csv \
    #     --auto_restrict_neuron_subset_file

    # python -u get_activations.py \
    #     --experiment_name top_fact_neurons \
    #     --experiment_type activation_subset \
    #     --feature_dataset wikidata_sorted_is_alive.pyth.128.6000 \
    #     --model "$model" \
    #     --neuron_subset_file top_fact_neurons.csv \
    #     --auto_restrict_neuron_subset_file

    # python -u get_activations.py \
    #     --experiment_name top_fact_neurons \
    #     --experiment_type activation_subset \
    #     --feature_dataset wikidata_sorted_occupation.pyth.128.6000 \
    #     --model "$model" \
    #     --neuron_subset_file top_fact_neurons.csv \
    #     --auto_restrict_neuron_subset_file

    # python -u get_activations.py \
    #     --experiment_name top_fact_neurons \
    #     --experiment_type activation_subset \
    #     --feature_dataset wikidata_sorted_political_party.pyth.128.3000 \
    #     --model "$model" \
    #     --neuron_subset_file top_fact_neurons.csv \
    #     --auto_restrict_neuron_subset_file

    # python -u get_activations.py \
    #     --experiment_name top_fact_neurons \
    #     --experiment_type activation_subset \
    #     --feature_dataset wikidata_sorted_sex_or_gender.pyth.128.6000 \
    #     --model "$model" \
    #     --neuron_subset_file top_fact_neurons.csv \
    #     --auto_restrict_neuron_subset_file

    python -u get_activations.py \
        --experiment_name top_fact_neurons \
        --experiment_type activation_subset \
        --feature_dataset distribution_id.pyth.512.-1 \
        --model "$model" \
        --neuron_subset_file top_fact_neurons.csv
done

