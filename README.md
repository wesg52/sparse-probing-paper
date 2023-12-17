# sparse-probing
Code repository for [Finding Neurons in a Haystack: Case Studies with Sparse Probing](https://arxiv.org/abs/2305.01610)

Pardon our mess. The basic core of sparse probing can be implemented very easily with just sklearn applied to a dataset of activations acquired with raw Pytorch hooks or TransformerLens. This repository is almost all experimental infrastructure and analysis specific to our set up of datasets and compute (slurm).

See [this](https://github.com/wesg52/llm-context-neurons/tree/main) repository for a minimal replication of finding context neurons.

## Organization
We expect most people to simply be interested in a large list of relevant neurons, available as CSVs within `interpretable_neurons/`. Note these are for the Pythia V0 models, which have since been updated on HuggingFace.

Our top level scripts for saving activations and running probing experiments can be count in `get_activations.py` and `probing_experiment.py`. All of command line argument configurations can be viewed in the `experiments/` directory, which contain all of the slurm scripts we used to run our experiments.

`probing_datasets/` contain the modules required to make and prepare all of our feature datasets. We recommend simply downloading them from [dropbox](https://www.dropbox.com/sh/cr2vw4owv6dkw7t/AADvkDwJYKyYDC56q1S8dAS_a?dl=0).

Analysis and plotting code is distributed within individual notebooks and `analysis/`.


## Instructions for reproducing
Note that our full experiments generate well over 1 TB of data and require substantial GPU and CPU time.

### Getting started
Create virtual environment and install required packages
```
git clone https://github.com/wesg52/sparse-probing-paper.git
cd sparse-probing
pip install virtualenv
python -m venv sparprob
source sparprob/bin/activate
pip install -r requirements.txt
```

Acquire Gurobi [license](https://www.gurobi.com/features/academic-named-user-license/). Free for academics. Make sure you are on campus wifi (you may also need to seperately install [grbgetkey](https://support.gurobi.com/hc/en-us/articles/360059842732)).

### Environment variables
To enable running our code in many different environments we use environemnt variables to specify the paths for all data input and output. For examples
```
export RESULTS_DIR=/Users/wesgurnee/Documents/mechint/sparse_probing/sparse-probing/results
export FEATURE_DATASET_DIR=/Users/wesgurnee/Documents/mechint/sparse_probing/sparse-probing/feature_datasets
export TRANSFORMERS_CACHE=/Users/wesgurnee/Documents/mechint/sparse_probing/sparse-probing/downloads
export HF_DATASETS_CACHE=/Users/wesgurnee/Documents/mechint/sparse_probing/sparse-probing/downloads
export HF_HOME=/Users/wesgurnee/Documents/mechint/sparse_probing/sparse-probing/downloads
```


## Cite us
If you found our work helpful, please cite our paper:
```
@article{gurnee2023finding,
  title={Finding Neurons in a Haystack: Case Studies with Sparse Probing},
  author={Gurnee, Wes and Nanda, Neel and Pauly, Matthew and Harvey, Katherine and Troitskii, Dmitrii and Bertsimas, Dimitris},
  journal={arXiv preprint arXiv:2305.01610},
  year={2023}
}
```
