#!/bin/bash
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o log/neurons.log-%j

# set environment variables
export PATH=$SPARSE_PROBING_ROOT:$PATH

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/home/gridsan/groups/maia_mechint/sparse_probing/results
export FEATURE_DATASET_DIR=/home/gridsan/groups/maia_mechint/sparse_probing/feature_datasets
export TRANSFORMERS_CACHE=/home/gridsan/groups/maia_mechint/models
export HF_DATASETS_CACHE=/home/gridsan/groups/maia_mechint/sparse_probing/hf_home
export HF_HOME=/home/gridsan/groups/maia_mechint/sparse_probing/hf_home
export INTERPRETABLE_NEURONS_DIR=$SPARSE_PROBING_ROOT/interpretable_neurons

sleep 0.1  # wait for paths to update

# activate environment and load modules
source $SPARSE_PROBING_ROOT/sparprob/bin/activate
source /etc/profile
module load gurobi/gurobi-1000

#PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b')
#n_layers:       6            12            24            16          24            32            32
#MODELS=('pythia-70m')
#MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b')
#MODELS=('pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b')

MODEL=pythia-6.9b
FEATURE=occupation # sex_or_gender is_alive occupation political_party occupation_athlete
N_SEQ=6000

PREFIX=wikidata_full
SUBSET_FILE=wikidata.csv

python get_activations.py \
    --experiment_type activation_subset \
    --feature_dataset "${PREFIX}_${FEATURE}.pyth.128.${N_SEQ}" \
    --model $MODEL \
    --neuron_subset_file $SUBSET_FILE \
    --auto_restrict_neuron_subset_file \
    --output_precision 32
    #--skip_computing_token_summary_df
