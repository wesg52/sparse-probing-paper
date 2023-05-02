#!/bin/bash
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o log/%j-wikidata-apd

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

PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b')
# 'pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b')
#n_layers:       6            12            24            16          24            32            32

for model in "${PYTHIA_MODELS[@]}"
do
    python -u get_activations.py \
        --experiment_name wikidata_apd \
        --experiment_type activation_probe_dataset \
        --feature_dataset wikidata_sorted_is_alive.pyth.128.6000 \
        --model $model \
        --activation_aggregation max

    python -u get_activations.py \
        --experiment_name wikidata_apd \
        --experiment_type activation_probe_dataset \
        --feature_dataset wikidata_sorted_occupation.pyth.128.6000 \
        --model $model \
        --activation_aggregation max

    python -u get_activations.py \
        --experiment_name wikidata_apd \
        --experiment_type activation_probe_dataset \
        --feature_dataset wikidata_sorted_occupation_athlete.pyth.128.5000 \
        --model $model \
        --activation_aggregation max

    python -u get_activations.py \
        --experiment_name wikidata_apd \
        --experiment_type activation_probe_dataset \
        --feature_dataset wikidata_sorted_political_party.pyth.128.3000 \
        --model $model \
        --activation_aggregation max

    python -u get_activations.py \
        --experiment_name wikidata_apd \
        --experiment_type activation_probe_dataset \
        --feature_dataset wikidata_sorted_sex_or_gender.pyth.128.6000 \
        --model $model \
        --activation_aggregation max
done
