import bz2
import json
import argparse
from tqdm import tqdm
import sys
import os

import torch
import numpy as np
import pandas as pd
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer

if __name__ == '__main__':
    sys.path.append(os.getenv('SPARSE_PROBING_ROOT'))
from probing_datasets.common import FeatureDataset
from config import FeatureDatasetConfig, ExperimentConfig
from load import load_raw_dataset

try:
    import re2 as re
except ImportError:
    import re


WIKIDATA_RAW_PATH = '/home/gridsan/groups/maia_mechint/datasets/wikidata/raw/latest-partial-10G.json.bz2'
#WIKIDATA_RAW_PATH = '/Users/mtp/Downloads/sparse-probing/datasets/wd/raw/latest-partial-10G.json.bz2'
PILE_TEST_PATH = os.path.join(
    os.environ['HF_DATASETS_CACHE'], 'pile-test.hf')

WIKIDATA_LOOKUP = {
    # properties
    'instance_of': 'P31',
    'sex_or_gender': 'P21',
    'occupation': 'P106',
    'date_of_birth': 'P569',
    'date_of_death': 'P570',
    'languages_spoken': 'P1412',
    'ethnic_group': 'P172',
    'country_of_citizenship': 'P27',
    'political_party': 'P102',
    # sex_or_gender
    'male': 'Q6581097',
    'female': 'Q6581072',
    # occupation
    'politician': 'Q82955',
    'actor': 'Q33999',
    'researcher': 'Q1650915',
    'singer': 'Q177220',
    'journalist': 'Q1930187',
    #'association football player': 'Q937857',  # TODO
    # occupation_athlete
    'association football player': 'Q937857',
    'basketball player': 'Q3665646',
    'American football player': 'Q19204627',
    'baseball player': 'Q10871364',
    'ice hockey player': 'Q11774891',
    # country_of_citizenship
    'United States of America': 'Q30',
    'United Kingdom': 'Q145',
    'Canada': 'Q16',
    'Australia': 'Q408',
    'France': 'Q142',
    'Germany': 'Q183',
    # political_party
    'Republican Party': 'Q29468',
    'Democratic Party': 'Q29552',
}
WIKIDATA_LOOKUP_INV = {v: k for k, v in WIKIDATA_LOOKUP.items()}

PROPERTY_CLASSES = {
    'sex_or_gender': [
        'female',
        'male',
    ],
    'occupation': [
        'politician',
        'actor',
        'researcher',
        'singer',
        'journalist',
        'athlete',
    ],
    'occupation_athlete': [
        'association football player',
        'basketball player',
        'American football player',
        'baseball player',
        'ice hockey player',
    ],
    'is_alive': [  # is_alive uses date of birth and death rather than property values
        'true',
        'false',
    ],
    'political_party': [
        'Republican Party',
        'Democratic Party',
    ],
    'country_of_citizenship': [
        'United States of America',
        'United Kingdom',
        'Canada',
        'Australia',
        'France',
        'Germany',
    ],
}

# TODO: this is a hack to get around the fact that we don't have a good way to
#       handle multiple properties for a single class
ATHLETE_OCCUPATIONS = [
    'Q2066131', 'Q172964', 'Q259327', 'Q476246', 'Q500097', 'Q549322',
    'Q584540', 'Q767703', 'Q810927', 'Q913532', 'Q969772', 'Q1385518',
    'Q1614023', 'Q2125610', 'Q2309784', 'Q2312865', 'Q2324660', 'Q2465611',
    'Q2540672', 'Q2568073', 'Q2570377', 'Q2730732', 'Q2749653', 'Q3388309',
    'Q3665646', 'Q3759615', 'Q3892689', 'Q3951423', 'Q4270517', 'Q4951095',
    'Q5465303', 'Q5610177', 'Q6060450', 'Q6665249', 'Q6885897', 'Q7623231',
    'Q7627813', 'Q7974325', 'Q9149093', 'Q10833314', 'Q10843263', 'Q10843402',
    'Q10871364', 'Q10873124', 'Q10889302', 'Q11124885', 'Q11292782', 'Q11296761',
    'Q11303721', 'Q11338576', 'Q11513337', 'Q11607585', 'Q11774891', 'Q11939963',
    'Q12039558', 'Q12299841', 'Q12369333', 'Q12803959', 'Q12840545', 'Q13141064',
    'Q13218361', 'Q13381863', 'Q13382355', 'Q13382487', 'Q13382519', 'Q13382533',
    'Q13382566', 'Q13382576', 'Q13388586', 'Q13414980', 'Q13415036', 'Q13464374',
    'Q13474373', 'Q13561328', 'Q13581129', 'Q14128148', 'Q15117302', 'Q15306067',
    'Q15972912', 'Q15982795', 'Q16004431', 'Q16004471', 'Q16278103', 'Q16947675',
    'Q17318006', 'Q17351861', 'Q17361156', 'Q17486376', 'Q17502714', 'Q17519504',
    'Q17524364', 'Q17611899', 'Q17619498', 'Q17682262', 'Q18199024', 'Q18437198',
    'Q18574233', 'Q18667447', 'Q18702210', 'Q18706712', 'Q18920523', 'Q19204627',
    'Q20751891', 'Q20965770', 'Q21057452', 'Q21081635', 'Q22695781', 'Q23308797',
    'Q23452173', 'Q23719050', 'Q23754399', 'Q23868946', 'Q23927336', 'Q24238692',
    'Q24573123', 'Q26111624', 'Q26233091', 'Q26326102', 'Q27062049', 'Q27062349',
    'Q27062494', 'Q27503001', 'Q27978698', 'Q28164897', 'Q28971125', 'Q29466935',
    'Q29474561', 'Q29579227', 'Q30175498', 'Q30339659', 'Q30525637', 'Q33135515',
    'Q38079409', 'Q41898139', 'Q43548373', 'Q43644099', 'Q50259293', 'Q51093963',
    'Q53645345', 'Q55962467', 'Q56164819', 'Q56677347', 'Q58546307', 'Q58823986',
    'Q58825429', 'Q59716278', 'Q60983289', 'Q61278580', 'Q63095188', 'Q63915541',
    'Q65240292', 'Q65697958', 'Q65924683', 'Q66759460', 'Q66839062', 'Q67164844',
    'Q67202689', 'Q71550599', 'Q72166732', 'Q72168361', 'Q78063080', 'Q86135347',
    'Q87252988', 'Q88109996', 'Q88191417', 'Q88203001', 'Q88209013', 'Q96864583',
    'Q97319426', 'Q97610399', 'Q98073294', 'Q98703899', 'Q98713221', 'Q101204362',
    'Q101500235', 'Q101542154', 'Q102068136', 'Q104218947', 'Q104548415', 'Q104737963',
    'Q105236846', 'Q105411957', 'Q105621122', 'Q106037358', 'Q106037372', 'Q106857893',
    'Q107392205', 'Q107407458', 'Q107639979', 'Q107690317', 'Q107921181', 'Q107932010',
    'Q108353052', 'Q108482633', 'Q108603213', 'Q108764079', 'Q109927927', 'Q110751065',
    'Q111017922', 'Q111036008', 'Q111045023', 'Q111304048', 'Q111655684', 'Q111729053',
    'Q112122383', 'Q112206577', 'Q112515016', 'Q112632134', 'Q114414171', 'Q115379459',
    'Q116085838', 'Q116445694', 'Q117134067', 'Q117321337', 'Q117393863', 'Q117450953',
    'Q117463084', 'Q201330', 'Q1317534', 'Q3909057', 'Q5575207', 'Q5575226',
    'Q11880349', 'Q61650479', 'Q1452497', 'Q3981532', 'Q22570303', 'Q111427270',
    'Q111427356', 'Q111427424', 'Q111429667', 'Q2191547', 'Q2376278', 'Q2412523',
    'Q2737671', 'Q3419806', 'Q3577354', 'Q3753218', 'Q60646581', 'Q78070424',
    'Q715772', 'Q979247', 'Q1989225', 'Q2665822', 'Q5036504', 'Q209897',
    'Q871529', 'Q1453373', 'Q7316299', 'Q11333808', 'Q20190513', 'Q30070907',
    'Q112225700', 'Q539274', 'Q632099', 'Q2178349', 'Q2328847', 'Q2455451',
    'Q3114868', 'Q5969982', 'Q15117395', 'Q15117415', 'Q19799599', 'Q52217314',
    'Q59384208', 'Q88479569', 'Q108886510', 'Q113486069', 'Q113996602', 'Q90904817',
    'Q846750', 'Q1365155', 'Q2668083', 'Q10701548', 'Q13381458', 'Q21517059',
    'Q24797688', 'Q26384038', 'Q88200618', 'Q101046010', 'Q45322469', 'Q19746576',
    'Q21423578', 'Q23067916', 'Q28870526', 'Q49904510', 'Q61707268', 'Q116057450',
    'Q107747791', 'Q3020534', 'Q4144610', 'Q11712888', 'Q12340540', 'Q18617021',
    'Q19801627', 'Q26869174', 'Q51208995', 'Q58851768', 'Q113510260', 'Q117667098',
    'Q379497', 'Q749387', 'Q20740414', 'Q28659301', 'Q252628', 'Q1852228',
    'Q13381376', 'Q23845879', 'Q819677', 'Q3411437', 'Q9292301', 'Q20683322',
    'Q59917804', 'Q109358584', 'Q113995668', 'Q18814798', 'Q25348407', 'Q48673045',
    'Q48676230', 'Q48995125', 'Q20900796', 'Q1866686', 'Q18715859', 'Q90496069',
    'Q105696682', 'Q107883088', 'Q112118895', 'Q115376577', 'Q174493', 'Q1048902',
    'Q1050571', 'Q1368170', 'Q11336312', 'Q55296227', 'Q105269', 'Q671382',
    'Q680271', 'Q2169760', 'Q20522910', 'Q30176225', 'Q108147149', 'Q25439032',
    'Q978854', 'Q1552832', 'Q1752130', 'Q2727289', 'Q9017214', 'Q29840095',
    'Q40871491', 'Q116193105', 'Q490253', 'Q2637418', 'Q388513', 'Q22269196',
    'Q13381428', 'Q13381689', 'Q13382122', 'Q13464497', 'Q13724897', 'Q13848274',
    'Q13854733', 'Q13856320', 'Q14605941', 'Q17405793', 'Q18510502', 'Q18534714',
    'Q21141381', 'Q26219487', 'Q35122846', 'Q51536424', 'Q51536572', 'Q62056380',
    'Q62627163', 'Q66604809', 'Q83169285', 'Q113374274', 'Q56885102', 'Q3186699',
    'Q11446504', 'Q48744240', 'Q2305269', 'Q7092294', 'Q16303386', 'Q1068064',
    'Q1773766', 'Q4009406', 'Q4439155', 'Q13381753', 'Q113621721', 'Q114050879',
    'Q117383481', 'Q179789', 'Q57749966', 'Q1519593', 'Q62056391', 'Q88191276',
    'Q88207871', 'Q117536057', 'Q56827013', 'Q1690874', 'Q5313922', 'Q14089670',
    'Q14373094', 'Q26237722', 'Q62056314', 'Q721834', 'Q17362882', 'Q65954523',
    'Q18664049', 'Q114239325', 'Q19841381', 'Q108320165', 'Q108730578', 'Q106647433',
    'Q117084683', 'Q109719129', 'Q117767136', 'Q16004432', 'Q16014296', 'Q47004511',
    'Q13381572', 'Q23892384', 'Q24037210', 'Q26831398', 'Q97487735', 'Q16029547',
    'Q56232598', 'Q108486512', 'Q5360816', 'Q9394993', 'Q13382981', 'Q13383011',
    'Q13388442', 'Q15709642', 'Q17516936', 'Q112037047', 'Q11244693', 'Q108887236',
    'Q378622', 'Q2385879', 'Q12327806', 'Q61043082', 'Q107568781', 'Q113172866',
    'Q43633947', 'Q26876991', 'Q100550234', 'Q58826204', 'Q63076925', 'Q937857',
    'Q117459552', 'Q1549466', 'Q247298', 'Q8193458', 'Q5705105', 'Q4661673',
    'Q10349745', 'Q12456639', 'Q64011604', 'Q88214604', 'Q108806370', 'Q39396',
    'Q10385084', 'Q93591280', 'Q1260222', 'Q7353388', 'Q193592', 'Q62056296',
    'Q107789510', 'Q108113346', 'Q772148', 'Q1198553', 'Q7659073', 'Q5931226',
    'Q47380712', 'Q4696968', 'Q87254054', 'Q87254201', 'Q117463046', 'Q1577970',
    'Q2017602', 'Q2859165', 'Q3320976', 'Q3392533', 'Q3392540', 'Q3392544',
    'Q3590981', 'Q3883921', 'Q5030708', 'Q6157423', 'Q7234596', 'Q10572447',
    'Q16670474', 'Q17504942', 'Q18697160', 'Q22704540', 'Q41696723', 'Q55383113',
    'Q55383242', 'Q111363586', 'Q117048858', 'Q3101194', 'Q11482903', 'Q117360769',
    'Q13382460', 'Q19827218', 'Q10866633', 'Q13219587', 'Q117661307', 'Q67020237',
    'Q108116154', 'Q28681386', 'Q113686041', 'Q113686258', 'Q21141393', 'Q21141408',
    'Q11598573', 'Q22916339', 'Q38370550', 'Q28836794', 'Q112073188', 'Q117479407',
    'Q28054240', 'Q107506477', 'Q116175001', 'Q62056361', 'Q88191594', 'Q88211956',
    'Q96656001', 'Q113510240', 'Q533097', 'Q650309', 'Q1237356', 'Q2380088',
    'Q3897311', 'Q4959392', 'Q7805499', 'Q16549481', 'Q20746016', 'Q21011245',
    'Q4333889', 'Q106807887', 'Q13382603', 'Q13382605', 'Q13382608', 'Q1201458',
    'Q6008848', 'Q8025128', 'Q11977073', 'Q16501245', 'Q18691898', 'Q90326494',
    'Q4672950', 'Q997419', 'Q3494356', 'Q6673520', 'Q6841695', 'Q7456982',
    'Q1067277', 'Q1423744', 'Q1424116', 'Q1937166', 'Q2166602', 'Q1137295',
    'Q3392545', 'Q15840574', 'Q23905927', 'Q749437', 'Q3014296', 'Q10841764',
    'Q10842936', 'Q12470043', 'Q55441806', 'Q55594106', 'Q56433160', 'Q98406729',
    'Q105748930', 'Q18200514', 'Q17361147', 'Q52161273', 'Q116480061', 'Q11293355',
    'Q24259627', 'Q55986349', 'Q63243629', 'Q106901236', 'Q108032642', 'Q109678900',
    'Q109678945', 'Q109679107', 'Q116236668', 'Q114358150', 'Q114358158', 'Q6054853',
    'Q114358123', 'Q114358125', 'Q659776', 'Q1229836', 'Q1323929', 'Q23906217',
    'Q23906220', 'Q23906224', 'Q23906225', 'Q23906226', 'Q23906228', 'Q23906229',
    'Q23906230', 'Q23906232', 'Q23906233', 'Q23906237', 'Q23906239', 'Q23906240',
    'Q23906243', 'Q23906248', 'Q23906251', 'Q23906254', 'Q23906263', 'Q23906266',
    'Q23906282', 'Q23906291', 'Q23906293', 'Q24068711',
]

MIN_NAME_LENGTH = 8
MAX_DEATH_YEAR = 1980


def build_json_file(n_lines, input_path, output_path, save_json=False):
    raw_file = bz2.BZ2File(input_path, 'r')
    raw_file.readline()  # skip first line

    output_json = []
    for _ in range(n_lines):
        line = raw_file.readline().strip()
        item = json.loads(line[:-1])
        output_json.append(item)

    with open(output_path + '.bz2', 'wb') as f:
        encoded_json = json.dumps(output_json).encode('utf-8')
        f.write(bz2.compress(encoded_json))

    if save_json:
        with open(output_path, 'w') as f:
            json.dump(output_json, f)


def get_property_values(item, property_id):
    if property_id in item['claims']:
        values = []
        for q in item['claims'][property_id]:
            try:
                value = q['mainsnak']['datavalue']['value']
                if 'id' in value:
                    values.append(value['id'])
                elif 'time' in value:
                    values.append(value['time'])
            except KeyError:
                pass
        return values
    else:
        return []


def get_n_properties(item):
    return len(item['claims'])


def make_re_list(items, re_size=500):
    re_list = []
    items_escaped = [re.escape(i) for i in items]
    for i in range(0, len(items), re_size):
        item_string = '|'.join(items_escaped[i:i+re_size])
        re_list.append(re.compile(item_string))
    return re_list


class WikidataFeatureDataset(FeatureDataset):

    def __init__(self):
        pass

    def prepare_dataset(self, exp_cfg: ExperimentConfig):
        '''
        Return valid indices and classes for the feature dataset
        '''

        dataset = self.load(exp_cfg.dataset_cfg)

        class_counts = {}
        for c in dataset['class']:
            if c not in class_counts:
                class_counts[c] = 0
            class_counts[c] += 1
        print(f'dataset size: {len(dataset)}, class breakdown: {class_counts}')

        # create feature datasets
        if exp_cfg.activation_aggregation is None or exp_cfg.activation_aggregation == 'none':
            feature_datasets = {}
            for c in class_counts.keys():
                # index is position within flattened (n_seq x seq_len,) array
                feature_indices = torch.Tensor(dataset['surname_index_end']).to(torch.int32)  # use surname by default
                feature_indices += torch.arange(len(feature_indices)) * \
                    len(dataset[0]['tokens'])

                # classes to {-1, +1}
                feature_classes = np.full(len(dataset), -1)
                feature_classes[[
                    c == example_class for example_class in dataset['class']]] = 1

                feature_datasets[c] = (feature_indices, feature_classes)
            return dataset, feature_datasets
        else:
            valid_index_mask = torch.zeros_like(dataset['tokens'])
            for ix, (start, end) in enumerate(zip(dataset['surname_index_start'], dataset['surname_index_end'])):
                valid_index_mask[ix, start:end+1] = 1

            feature_datasets = {}
            for c in class_counts.keys():
                # classes to {-1, +1}
                feature_classes = np.full(len(dataset), -1)
                feature_classes[[
                    c == example_class for example_class in dataset['class']]] = 1

                feature_datasets[c] = (valid_index_mask, feature_classes)
            return dataset, feature_datasets

    def make(
        self,
        dataset_config: FeatureDatasetConfig,
        args: dict,
        table: Dataset,
        text_dataset: Dataset,
        tokenizer: AutoTokenizer,
        cache=True,
        num_proc=1,
    ) -> Dataset:
        '''
        Arguments:
            args: command line arguments from make_feature_datasets
            table: wikidata entries
            text_dataset: dataset of raw strings (same as was used to create the table)

        Returns feature_dataset with columns:
            text: raw strings
            tokens: tokenized strings of consitent length
            name: name of person
            name_index_start: index of the start of the name in tokens
            name_index_end: index of the end of the name in tokens
            surname_index_start: index of the start of the surname in tokens
            surname_index_end: index of the end of the surname in tokens
            class: property class for the example
        '''

        if 'wikidata_property' not in args.keys():
            raise ValueError('Must provide "wikidata_property" argument for a wikidata feature dataset')
        if args['wikidata_property'] not in PROPERTY_CLASSES.keys():
            raise ValueError(f'Invalid wikidata property: "{args["wikidata_property"]}"')
        wikidata_property = args['wikidata_property']

        # sort the dataset by n_properties to include (approximately) most well-known names
        table = table.sort('n_properties', reverse=True)

        # limit number of name repeats in the dataset
        max_per_class = -1 if 'max_per_class' not in args.keys() else args['max_per_class']
        if max_per_class > 0:  # account for multiple processes, messy but fine in practice
            max_per_class //= num_proc
        max_name_repeats = args['max_name_repeats']

        # filter names for pile repeats
        min_pile_repeats = args['min_pile_repeats']
        if min_pile_repeats > 1:
            table_df = pd.DataFrame(table)
            name_counts = table_df.groupby('name').count()
            valid_names = list(name_counts[name_counts['dataset_index'] >= min_pile_repeats].index)

        class_counts = {c: 0 for c in PROPERTY_CLASSES[wikidata_property]}
        name_counts = {}

        cols = ['text', 'tokens', 'name', 'name_index_start', 'name_index_end',
                'surname_index_start', 'surname_index_end', 'class']

        def tokenize(batched_example):
            # don't need to compute anything if we have enough sequences
            if dataset_config.n_sequences > 0 and sum(class_counts.values()) >= dataset_config.n_sequences:
                return {c: [] for c in cols}

            example = {k: v[0] for k, v in batched_example.items()}
            name = example['name']

            # filter based on repeats
            if name in name_counts.keys() and name_counts[name] >= max_name_repeats:
                return {c: [] for c in cols}
            # filter names for pile repeats
            if min_pile_repeats > 1 and name not in valid_names:
                return {c: [] for c in cols}

            properties = {}
            for k, v in example.items():
                if k not in ['name', 'line_index', 'id', 'dataset_index', 'match_position',
                             'n_properties']:
                    properties[k] = json.loads(v.replace("'", '"'))

            # need to handle some properties separately
            if wikidata_property == 'is_alive':
                has_birth = len(properties['date_of_birth']) > 0
                has_death = len(properties['date_of_death']) > 0
                if has_death:
                    death_year = int(properties['date_of_death'][0][:5])
                    if death_year > MAX_DEATH_YEAR:
                        return {c: [] for c in cols}

                if has_birth and not has_death:
                    example_classes = ['true']
                elif has_death:
                    example_classes = ['false']
                else:
                    return {c: [] for c in cols}
            elif wikidata_property == 'political_party':
                example_classes = [WIKIDATA_LOOKUP_INV[p] for p in properties[wikidata_property]
                                   if p in WIKIDATA_LOOKUP_INV.keys() and
                                   WIKIDATA_LOOKUP_INV[p] in PROPERTY_CLASSES[wikidata_property]]

                # filter for people who are politicians
                if WIKIDATA_LOOKUP['politician'] not in properties['occupation']:
                    return {c: [] for c in cols}
            elif wikidata_property == 'occupation_athlete':
                example_classes = [WIKIDATA_LOOKUP_INV[p] for p in properties['occupation']
                                   if p in WIKIDATA_LOOKUP_INV.keys() and
                                   WIKIDATA_LOOKUP_INV[p] in PROPERTY_CLASSES['occupation_athlete']]
            elif wikidata_property == 'occupation':
                example_classes = [WIKIDATA_LOOKUP_INV[p] for p in properties[wikidata_property]
                                   if p in WIKIDATA_LOOKUP_INV.keys() and
                                   WIKIDATA_LOOKUP_INV[p] in PROPERTY_CLASSES[wikidata_property]]

                # check for generic athlete
                for p in properties[wikidata_property]:
                    if p in ATHLETE_OCCUPATIONS:
                        example_classes.append('athlete')
                        break
            else:
                example_classes = [WIKIDATA_LOOKUP_INV[p] for p in properties[wikidata_property]
                                   if p in WIKIDATA_LOOKUP_INV.keys() and
                                   WIKIDATA_LOOKUP_INV[p] in PROPERTY_CLASSES[wikidata_property]]

            # filter based on class
            if max_per_class > 0:
                example_classes = [c for c in example_classes if class_counts[c] < max_per_class]
            if len(example_classes) != 1:
                return {c: [] for c in cols}

            for c in example_classes:
                class_counts[c] += 1

            full_text = text_dataset[example['dataset_index']
                                    ]['text'][:example['match_position'] + len(name)]

            full_text_tokens = tokenizer(full_text).input_ids
            name_tokens = tokenizer(name).input_ids
            name_tokens_space = tokenizer(' ' + name).input_ids

            tokens = full_text_tokens[-args['seq_len'] +
                                      1 if args['add_bos'] else 0:]
            if args['add_bos']:
                tokens = [tokenizer.bos_token_id] + tokens

            if name_tokens == tokens[-len(name_tokens):]:
                name_index_start = len(tokens) - len(name_tokens)
            elif name_tokens_space == tokens[-len(name_tokens_space):]:
                name_index_start = len(tokens) - len(name_tokens_space)
            else:
                return {c: [] for c in cols}
            name_index_end = len(tokens) - 1

            # store the surname index
            surname = ' '.join(name.split(' ')[1:])
            surname_tokens = tokenizer(surname).input_ids
            surname_tokens_space = tokenizer(' ' + surname).input_ids
            if surname_tokens == tokens[-len(surname_tokens):]:
                surname_index_start = len(tokens) - len(surname_tokens)
            elif surname_tokens_space == tokens[-len(surname_tokens_space):]:
                surname_index_start = len(tokens) - len(surname_tokens_space)
            else:
                return {c: [] for c in cols}
            surname_index_end = name_index_end

            if len(tokens) < args['seq_len']:
                tokens = tokens + (args['seq_len'] -
                                   len(tokens)) * [tokenizer.pad_token_id]
            assert len(tokens) == args['seq_len']

            text = ''.join(tokenizer.decode(tokens[1:])) if args['add_bos'] else ''.join(
                tokenizer.decode(tokens))

            # track repeats
            # NOTE: we can go over max_name_repeats in some cases but this shouldn't matter
            if name not in name_counts:
                name_counts[name] = len(example_classes)
            else:
                name_counts[name] += len(example_classes)

            return {
                    'text': [text] * len(example_classes),
                    'tokens': [tokens] * len(example_classes),
                    'name': [name] * len(example_classes),
                    'name_index_start': [name_index_start] * len(example_classes),
                    'name_index_end': [name_index_end] * len(example_classes),
                    'surname_index_start': [surname_index_start] * len(example_classes),
                    'surname_index_end': [surname_index_end] * len(example_classes),
                    'class': example_classes,
                    }

        print('tokenizing table strings...')
        # NOTE: we need to keep batch_size=1
        cols_to_remove = list(set(table.column_names) - set(cols))
        feature_dataset = table.map(
                tokenize, num_proc=num_proc, batched=True, batch_size=1,
                remove_columns=cols_to_remove)

        # clean up and save
        feature_dataset = feature_dataset.filter(lambda example:
                'tokens' in example.keys() and example['tokens'] is not None)

        if max_per_class <= 0:  # only care about the total number of sequences if we're not already limiting by class
            if dataset_config.n_sequences > 0 and len(feature_dataset) > dataset_config.n_sequences:
                feature_dataset = feature_dataset.select(
                    range(dataset_config.n_sequences))
        feature_dataset.set_format(type="torch", columns=[
                                   'tokens'], output_all_columns=True)

        print('Produced feature dataset:')
        print(feature_dataset)

        for c in PROPERTY_CLASSES[wikidata_property]:
            s = [1 for example_c in feature_dataset['class'] if example_c == c]
            print(f'{c}: {sum(s)}')

        if cache:
            print('Saving dataset...')
            self.save(dataset_config, feature_dataset)

        return feature_dataset


if __name__ == '__main__':
    '''
    Create a table from the raw database. Writes a table to a csv containing:
        line_index: the line index of the entity from the wikidata dump (may be repeated)
        id: wikidata entity name of the person (may be repeated)
        name: name of the person (may be repeated)
            (other columns for features)
        dataset_index: index of the row of the dataset containing the name match
        match_position: index of the name match within the specified dataset string
    '''

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--n_lines', type=int, default=1_000_000)
    parser.add_argument('-r', '--raw_path', type=str,
                        default=WIKIDATA_RAW_PATH)
    parser.add_argument('-d', '--dataset_path', type=str,
                        default=PILE_TEST_PATH)
    parser.add_argument('-o', '--output_path', type=str,
                        default='./wikidata.csv')
    args = parser.parse_args()

    print('Searching wikidata dump...')
    raw_file = bz2.BZ2File(args.raw_path, 'r')
    raw_file.readline()  # skip first line

    name_lookup = {}
    for lix in tqdm(range(args.n_lines if args.n_lines > 0 else 999999999999999999)):
        # break at end of file
        try:
            line = raw_file.readline().strip()
        except EOFError:
            print('Reached end of file')
            break
        try:
            item = json.loads(line[:-1])
        except json.decoder.JSONDecodeError:
            break

        instance_of = get_property_values(item, WIKIDATA_LOOKUP['instance_of'])
        if 'Q5' not in instance_of:  # instance of human
            continue
        try:
            name = item['labels']['en']['value']
            properties = {'line_index': lix,
                          'id': item['id'], 'name': name}
        except KeyError:  # sometimes entries don't have an english label
            continue

        # filter
        if len(name) < MIN_NAME_LENGTH:  # length
            continue
        if ' ' not in name:  # contains space
            continue

        # check for duplicate names
        n_properties = get_n_properties(item)
        if name in name_lookup.keys():
            # use the heuristic of choosing the entity with the most properties
            if n_properties <= name_lookup[name][1]:
                continue

        # get properties
        properties['sex_or_gender'] = get_property_values(
            item, WIKIDATA_LOOKUP['sex_or_gender'])
        properties['occupation'] = get_property_values(
            item, WIKIDATA_LOOKUP['occupation'])
        properties['date_of_birth'] = get_property_values(
            item, WIKIDATA_LOOKUP['date_of_birth'])
        properties['date_of_death'] = get_property_values(
            item, WIKIDATA_LOOKUP['date_of_death'])
        properties['languages_spoken'] = get_property_values(
            item, WIKIDATA_LOOKUP['languages_spoken'])
        properties['ethnic_group'] = get_property_values(
            item, WIKIDATA_LOOKUP['ethnic_group'])
        properties['country_of_citizenship'] = get_property_values(
            item, WIKIDATA_LOOKUP['country_of_citizenship'])
        properties['political_party'] = get_property_values(
            item, WIKIDATA_LOOKUP['political_party'])
        properties['n_properties'] = n_properties

        name_lookup[name] = (properties, n_properties)

    # search pile and include sequence info in the table
    text_dataset = load_raw_dataset(args.dataset_path)
    print('Searching dataset for matching strings...')
    re_list = make_re_list(list(name_lookup.keys()))

    text_re = []
    for i, text in enumerate(tqdm(text_dataset['text'])):
        example_matches = []
        for j, r in enumerate(re_list):
            result = r.search(text)
            if result is not None:
                example_matches.append({
                    'name': text[result.start():result.end()],
                    'dataset_index': i,
                    'match_position': result.start(),
                })

        # handle name substrings
        example_matches.sort(key=lambda x: 1_000_000 * x['match_position'] + len(x['name']))
        to_drop = []
        for i, (a, b) in enumerate(zip(example_matches[:-1], example_matches[1:])):
            if a['match_position'] == b['match_position']:
                to_drop.append(i)
        for i in to_drop[::-1]:
            del example_matches[i]

        text_re.extend(example_matches)

    # make a table
    table_list = [{**name_lookup[match['name']][0], **match} for match in text_re]

    df = pd.DataFrame(table_list)
    df.to_csv(args.output_path, index=False)
    print(f'wrote {len(df)} entries to {args.output_path}')
    print(f'final file position: {raw_file.tell()}')
