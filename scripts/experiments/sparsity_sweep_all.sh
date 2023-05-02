#!/bin/bash
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -o log/%j-%a-full_sparsity_sweep_final.log
#SBATCH -a 1-32


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

PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b')


for model in "${PYTHIA_MODELS[@]}"
do
    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --model "$model" \
        --feature_dataset programming_lang_id.pyth.512.-1 \
        --activation_aggregation mean \
        --normalize_activations \
        --seed 42 \
        --save_features_together


    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --model "$model" \
        --feature_dataset distribution_id.pyth.512.-1 \
        --activation_aggregation mean \
        --normalize_activations \
        --seed 42 \
        --save_features_together


    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --model "$model" \
        --feature_dataset natural_lang_id.pyth.512.-1 \
        --activation_aggregation mean \
        --normalize_activations \
        --seed 42 \
        --save_features_together


    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --model "$model" \
        --feature_dataset compound_words.pyth.24.-1 \
        --normalize_activations \
        --seed 42 \
        --save_features_together


    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --model "$model" \
        --feature_dataset text_features.pyth.256.10000 \
        --normalize_activations \
        --seed 42 \
        --save_features_together


    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --model "$model" \
        --feature_dataset ewt.pyth.512.-1 \
        --normalize_activations \
        --seed 42 \
        --save_features_together


    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --model "$model" \
        --feature_dataset latex.pyth.1024.-1 \
        --normalize_activations \
        --seed 42 \
        --save_features_together


    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --feature_dataset wikidata_sorted_is_alive.pyth.128.6000 \
        --model $model \
        --activation_aggregation max \
        --normalize_activations \
        --seed 42 \
        --save_features_together

    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --feature_dataset wikidata_sorted_occupation.pyth.128.6000 \
        --model $model \
        --activation_aggregation max \
        --normalize_activations \
        --seed 42 \
        --save_features_together

    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --feature_dataset wikidata_sorted_occupation_athlete.pyth.128.5000 \
        --model $model \
        --activation_aggregation max \
        --normalize_activations \
        --seed 42 \
        --save_features_together

    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --feature_dataset wikidata_sorted_political_party.pyth.128.3000 \
        --model $model \
        --activation_aggregation max \
        --normalize_activations \
        --seed 42 \
        --save_features_together

    python -u probing_experiment.py \
        --experiment_name full_sparsity_sweep_final \
        --experiment_type rotation_baseline_dxd telescopic_sparsity_sweep \
        --feature_dataset wikidata_sorted_sex_or_gender.pyth.128.6000 \
        --model $model \
        --activation_aggregation max \
        --normalize_activations \
        --seed 42 \
        --save_features_together
done