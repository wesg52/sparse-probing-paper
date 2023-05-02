#!/bin/bash
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -o log/%j-%a-osp_tuning.log
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


python probing_experiment.py \
    --experiment_name osp_hparam_tuning \
    --experiment_type osp_tuning \
    --model pythia-1b \
    --feature_dataset programming_lang_id.pyth.512.-1 \
    --activation_aggregation mean \
    --osp_upto_k 8 \
    --normalize_activations \
    --gurobi_timeout 90

python probing_experiment.py \
    --experiment_name osp_hparam_tuning \
    --experiment_type osp_tuning \
    --model pythia-6.9b \
    --feature_dataset distribution_id.pyth.512.-1 \
    --activation_aggregation mean \
    --osp_upto_k 8 \
    --normalize_activations \
    --gurobi_timeout 90

python probing_experiment.py \
    --experiment_name osp_hparam_tuning \
    --experiment_type osp_tuning \
    --model pythia-410m \
    --feature_dataset text_features.pyth.256.10000 \
    --osp_upto_k 8 \
    --normalize_activations \
    --gurobi_timeout 90

python probing_experiment.py \
    --experiment_name osp_hparam_tuning \
    --experiment_type osp_tuning \
    --model pythia-160m \
    --feature_dataset ewt.pyth.512.-1 \
    --osp_upto_k 8 \
    --normalize_activations \
    --feature_subset morph \
    --gurobi_timeout 60


python probing_experiment.py \
    --experiment_name osp_hparam_tuning \
    --experiment_type osp_tuning \
    --model pythia-1.4b \
    --feature_dataset latex.pyth.1024.-1 \
    --osp_upto_k 8 \
    --normalize_activations \
    --gurobi_timeout 90


python probing_experiment.py \
    --experiment_name osp_hparam_tuning \
    --experiment_type osp_tuning \
    --model pythia-2.8b \
    --feature_dataset compound_words.pyth.24.-1 \
    --osp_upto_k 8 \
    --normalize_activations \
    --feature_subset social-security,trial-court,mental-health \
    --gurobi_timeout 60