#!/bin/bash
#SBATCH -c 8
#SBATCH -o log/table.log-%j-%a

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

N_LINES=-1
#N_LINES=250

# create the table
python probing_datasets/wikidata.py \
    --n_lines $N_LINES \
    --raw_path /home/gridsan/groups/maia_mechint/datasets/wikidata/raw/latest-all.json.bz2 \
    --dataset_path /home/gridsan/groups/maia_mechint/datasets/pile-test.hf \
    --output_path "/home/gridsan/groups/maia_mechint/datasets/wikidata/wikidata_pile_test_${N_LINES}.csv"
