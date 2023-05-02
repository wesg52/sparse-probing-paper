#!/bin/bash
#SBATCH -c 4
#SBATCH -o log/dataset.log-%j-%a

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

PROPERTY=political_party  # sex_or_gender is_alive occupation political_party occupation_athlete
MAX_PER_CLASS=500
N_SEQS=1000
TABLE_SIZE=-1
NUM_PROC=1  # probably should keep this at 1 unless max_name_repeats doesn't matter
MAX_NAME_REPEATS=1  # note that this is per process
MIN_PILE_REPEATS=2

PREFIX=wikidata_full

# create the table
python make_feature_datasets.py \
    --feature_collection wikidata \
    --model pythia-70m \
    --seq_len 128 \
    --n_seqs $N_SEQS \
    --num_proc $NUM_PROC \
    --wikidata_table_path "/home/gridsan/groups/maia_mechint/datasets/wikidata/wikidata_pile_test_${TABLE_SIZE}.csv" \
    --dataset_name "${PREFIX}_${PROPERTY}" \
    --wikidata_property $PROPERTY \
    --max_per_class $MAX_PER_CLASS \
    --max_name_repeats $MAX_NAME_REPEATS \
    --min_pile_repeats $MIN_PILE_REPEATS
