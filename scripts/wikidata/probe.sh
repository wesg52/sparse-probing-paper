#!/bin/bash
#SBATCH -c 20
#SBATCH -o log/probe.log-%j-%a
#SBATCH -a 1-8

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

#PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b')
#n_layers:       6            12            24            16          24            32            32

MODELS=('pythia-6.9b')

FEATURE=occupation
N_SEQ=6000

PREFIX=wikidata_full

for model in "${MODELS[@]}"
do
    python probing_experiment.py \
        --experiment_name wikidata/$FEATURE \
        --model "$model" \
        --feature_dataset "${PREFIX}_${FEATURE}.pyth.128.${N_SEQ}" \
        --experiment_type telescopic_sparsity_sweep enumerate_monosemantic \
        --activation_aggregation max \
        --normalize_activations
done
