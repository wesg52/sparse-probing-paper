import os
import argparse

from datasets import Dataset
import pandas as pd

from probing_datasets import token_supervised
from probing_datasets import language_id
from probing_datasets import counterfact
from probing_datasets import distribution_id
from probing_datasets import ngrams
from probing_datasets import pile_test
from probing_datasets import neuron_stimulus
from probing_datasets import ewt
from probing_datasets import wikidata
from probing_datasets import position
from probing_datasets import latex

from config import *
from load import *


GITHUB_DATASET_PATH = os.path.join(
    os.environ['HF_DATASETS_CACHE'], 'the_pile', 'github', 'github_lang_id.hf')
EUROPARL_DATASET_PATH = os.path.join(
    os.environ['HF_DATASETS_CACHE'], 'the_pile', 'europarl.hf')
PILE_SUBSET_PATH = os.path.join(
    os.environ['HF_DATASETS_CACHE'], 'NeelNanda/pile-10k', 'train')
COUNTERFACT_DATASET_PATH = os.path.join(
    os.environ['HF_DATASETS_CACHE'], 'NeelNanda/counterfact-tracing', 'train')
PILE_EVEN_SPLITS_PATH = os.path.join(
    os.environ['HF_DATASETS_CACHE'], 'test_subset_with_even_slices.hf')
#PILE_TEST_PATH = os.path.join(
#    os.environ['HF_DATASETS_CACHE'], 'pile-test.hf')
PILE_TEST_PATH = '/home/gridsan/groups/maia_mechint/datasets/pile-test.hf'
PREPROCESSED_EWT_PATH = os.path.join(
    os.environ['HF_DATASETS_CACHE'], 'preprocessed_ewt_512.hf')
WIKIDATA_TABLE_PATH = '/home/gridsan/groups/maia_mechint/datasets/wikidata/wikidata_pile_test_1000000.csv'
ARXIV_SUBSET_PATH = os.path.join(
    os.environ['HF_DATASETS_CACHE'], 'arxiv_dataset.hf')


TEXT_FEATURES = {
    'text_features': token_supervised.true_binary_token_supervised_feature_conditions,
    'has_suffix': token_supervised.suffix_conditions,
    'has_prefix': token_supervised.prefix_conditions
}


def prepare_feature_dataset(exp_cfg):
    feature_collection = exp_cfg.dataset_cfg.dataset_name
    if feature_collection in TEXT_FEATURES:
        ptfd = token_supervised.PlainTextFeatureDataset(
            feature_collection, TEXT_FEATURES[feature_collection])
        tokenized_dataset, feature_datasets = ptfd.prepare_dataset(exp_cfg)

    elif feature_collection == 'programming_lang_id':
        plfd = language_id.LanguageIDFeatureDataset(
            'programming_lang_id', language_id.CODE_LANGS)
        tokenized_dataset, feature_datasets = plfd.prepare_dataset(exp_cfg)

    elif feature_collection == 'natural_lang_id':
        nlfd = language_id.LanguageIDFeatureDataset(
            'natural_lang_id', language_id.TOP_NATURAL_LANGS)
        tokenized_dataset, feature_datasets = nlfd.prepare_dataset(exp_cfg)

    elif feature_collection == 'distribution_id':
        ddfd = distribution_id.DataDistributionIDFeatureDataset(
            'distribution_id', distribution_id.DATASET_SPLITS)
        tokenized_dataset, feature_datasets = ddfd.prepare_dataset(exp_cfg)

    elif feature_collection == 'compound_words':
        cwfd = ngrams.BigramFeatureDataset(
            'compound_words', ngrams.COMPOUND_WORDS)
        tokenized_dataset, feature_datasets = cwfd.prepare_dataset(exp_cfg)

    elif feature_collection == 'counterfact':
        cffd = counterfact.CounterfactFeatureDataset()
        tokenized_dataset, feature_datasets = cffd.prepare_dataset(exp_cfg)

    elif feature_collection == 'ewt':
        ewtds = ewt.LinguisticFeatureDataset('ewt')
        tokenized_dataset, feature_datasets = ewtds.prepare_dataset(exp_cfg)

    elif 'wikidata' in feature_collection:  # store more information in the dataset type
        wdfd = wikidata.WikidataFeatureDataset()
        tokenized_dataset, feature_datasets = wdfd.prepare_dataset(exp_cfg)

    elif feature_collection == 'position':
        pds = position.PositionFeatureDataset()
        tokenized_dataset, feature_datasets = pds.prepare_dataset(exp_cfg)

    elif feature_collection == 'latex':
        lfd = latex.LatexFeatureDataset()
        tokenized_dataset, feature_datasets = lfd.prepare_dataset(exp_cfg)

    else:
        raise ValueError('Invalid feature_dataset type')

    if int(exp_cfg.dataset_cfg.n_sequences) == -1:
        exp_cfg.dataset_cfg.n_sequences = len(tokenized_dataset)
    return tokenized_dataset, feature_datasets


def make_token_supervised_feature_datasets(args, model, dataset_config):
    raw_dataset = load_raw_dataset(
        PILE_EVEN_SPLITS_PATH, dataset_config.n_sequences)
    feature_collection = args['feature_collection']
    ptfd = token_supervised.PlainTextFeatureDataset(
        feature_collection, TEXT_FEATURES[feature_collection])
    ptfd.make(dataset_config, args, raw_dataset, model.tokenizer)


def make_programming_lang_id(args, model, dataset_config):
    raw_dataset = load_raw_dataset(
        GITHUB_DATASET_PATH, dataset_config.n_sequences)
    plfd = language_id.LanguageIDFeatureDataset(
        'programming_lang_id', language_id.CODE_LANGS)
    plfd.make(dataset_config, args, raw_dataset, model.tokenizer)


def make_natural_lang_id(args, model, dataset_config):
    raw_dataset = load_raw_dataset(
        EUROPARL_DATASET_PATH, dataset_config.n_sequences)
    nlfd = language_id.LanguageIDFeatureDataset(
        'natural_lang_id', language_id.TOP_NATURAL_LANGS)
    nlfd.make(dataset_config, args, raw_dataset, model.tokenizer)


def make_counterfact(args, model, dataset_config):
    raw_dataset = load_raw_dataset(
        COUNTERFACT_DATASET_PATH, dataset_config.n_sequences)
    cffd = counterfact.CounterfactFeatureDataset()
    cffd.make(dataset_config, args, raw_dataset, model.tokenizer)


def make_distribution_id(args, model, dataset_config):
    raw_dataset = load_raw_dataset(
        PILE_EVEN_SPLITS_PATH, dataset_config.n_sequences)
    ddfd = distribution_id.DataDistributionIDFeatureDataset(
        'distribution_id', distribution_id.DATASET_SPLITS)
    ddfd.make(dataset_config, args, raw_dataset, model.tokenizer)


def make_compound_words_ds(args, model, dataset_config):
    raw_dataset = load_raw_dataset(
        PILE_TEST_PATH, dataset_config.n_sequences)
    bffd = ngrams.BigramFeatureDataset('compound_words', ngrams.COMPOUND_WORDS)
    bffd.make(dataset_config, args, raw_dataset, model.tokenizer)


def make_pile_test_ds(args, model, dataset_config):
    raw_dataset = load_raw_dataset(
        PILE_TEST_PATH, dataset_config.n_sequences)
    ptfd = pile_test.PileTestSplitFeatureDataset('pile_test')
    ptfd.make(dataset_config, args, raw_dataset, model.tokenizer)


def make_position_ds(args, model, dataset_config):
    raw_dataset = load_raw_dataset(
        PILE_TEST_PATH, dataset_config.n_sequences)
    posfd = position.PositionFeatureDataset()
    posfd.make(dataset_config, args, raw_dataset, model.tokenizer)


def make_neuron_stimilus_ds(args, model, dataset_config):
    tokenized_dataset = load_feature_dataset(
        'pile_test.pyth.512.-1', dataset_config.n_sequences)
    # TODO: make stimulus dict a parameter
    nsfd = neuron_stimulus.NeuronStimulusFeatureDataset(
        'neuron_stimulus', neuron_stimulus.PYTHIA_70M_L1_N111_STIMULI)
    nsfd.make(dataset_config, args, tokenized_dataset)


def make_ewt_ds(args, model, dataset_config):
    # if not os.path.exists(PREPROCESSED_EWT_PATH):
    ewt.make_preprocessed_ewt_dataset(model, dataset_config.ctx_len)
    preprocess_ewt_ds = load_raw_dataset(
        PREPROCESSED_EWT_PATH, dataset_config.n_sequences)
    ewtfd = ewt.LinguisticFeatureDataset('ewt')
    ewtfd.make(dataset_config, args, preprocess_ewt_ds)


def make_wikidata(args, model, dataset_config):
    print('Loading wikidata table...')
    df = pd.read_csv(args['wikidata_table_path'])
    table = Dataset.from_pandas(df)
    print('Loading raw dataset...')
    pile_test = load_raw_dataset(PILE_TEST_PATH)
    #pile_test = load_raw_dataset('/Users/mtp/Downloads/sparse-probing/datasets/pile-test.hf')
    cffd = wikidata.WikidataFeatureDataset()
    cffd.make(dataset_config, args, table, pile_test, model.tokenizer, num_proc=args['num_proc'])


def make_latex(args, model, dataset_config):
    raw_dataset = load_raw_dataset(
        ARXIV_SUBSET_PATH, dataset_config.n_sequences)
    lfd = latex.LatexFeatureDataset()
    lfd.make(dataset_config, args, raw_dataset, model.tokenizer)


if __name__ == '__main__':
    # Each dataset has a default config; use command line arguments to select the
    # the dataset and change the default config.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--feature_collection', required=True, help='Name of the feature collection to make')
    parser.add_argument(
        '-m', '--model', default='pythia-70m', help='Name of model from TransformerLens')
    parser.add_argument(
        '-d', '--dataset_name', help='Save name of dataset, defaults to feature_collection')
    parser.add_argument(
        '-l', '--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument(
        '-n', '--n_seqs', type=int, default=-1, help='Number of sequences to save (-1 for all)')
    parser.add_argument(
        '-b', '--add_bos', default=True, type=bool, help='Add bos token')
    parser.add_argument(
        '--dataset_size', type=int, help='Number of probing indices to use')
    parser.add_argument(
        '--target_positive_fraction', type=float, help='Target fraction of positive examples')
    parser.add_argument(
        '--min_positive_fraction', type=float, help='Minimum fraction of positive examples, otherwise ignore')
    parser.add_argument(
        '--ignore_first_k', type=int, help='Ignore first k tokens in sequence')
    parser.add_argument(
        '--lang_id_n_tokens', type=int, help='Number of tokens to use for language id datasets')
    parser.add_argument(
        '--wikidata_min_name_length', type=int, default=8)
    parser.add_argument(
        '--random_seed', default=1, type=int, help='Random seed for reproducibility')
    parser.add_argument(
        '--num_proc', default=4, type=int, help='Number of processes to use for dataset creation')
    # wikidata-specific
    parser.add_argument(
        '--wikidata_table_path', type=str, default=WIKIDATA_TABLE_PATH)
    parser.add_argument(
        '--wikidata_property', type=str, help='Wikidata property to probe')
    parser.add_argument(
        '--max_per_class', type=int, default=-1, help='Maximum number of examples per class')
    parser.add_argument(
        '--max_name_repeats', type=int, default=1, help='Maximum number of repeats of each name')
    parser.add_argument(
        '--min_pile_repeats', type=int, default=1, help='Filter names by number of times they appear in the pile test dataset')

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    dataset_config = FeatureDatasetConfig(
        dataset_name=args.get('dataset_name', args['feature_collection']),
        tokenizer_name=args.get('model')[:4],
        ctx_len=args.get('seq_len'),
        n_sequences=args.get('n_seqs'),
    )

    model = load_model(args['model'])

    if args['feature_collection'] in TEXT_FEATURES:
        make_token_supervised_feature_datasets(args, model, dataset_config)

    elif args['feature_collection'] == 'programming_lang_id':
        make_programming_lang_id(args, model, dataset_config)

    elif args['feature_collection'] == 'natural_lang_id':
        make_natural_lang_id(args, model, dataset_config)

    elif args['feature_collection'] == 'counterfact':
        make_counterfact(args, model, dataset_config)

    elif args['feature_collection'] == 'distribution_id':
        make_distribution_id(args, model, dataset_config)

    elif args['feature_collection'] == 'compound_words':
        make_compound_words_ds(args, model, dataset_config)

    elif args['feature_collection'] == 'pile_test':
        make_pile_test_ds(args, model, dataset_config)

    elif args['feature_collection'] == 'neuron_stimulus':
        make_neuron_stimilus_ds(args, model, dataset_config)

    elif args['feature_collection'] == 'ewt':
        make_ewt_ds(args, model, dataset_config)

    elif args['feature_collection'] == 'wikidata':
        make_wikidata(args, model, dataset_config)

    elif args['feature_collection'] == 'position':
        make_position_ds(args, model, dataset_config)

    elif args['feature_collection'] == 'latex':
        make_latex(args, model, dataset_config)

    else:
        raise ValueError(
            f'Unknown feature collection: {args["feature_collection"]}')
