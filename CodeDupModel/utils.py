import glob
import json
import os
import datasets
import pandas as pd
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def load_labels_map():
    _labels_map = {}
    if os.path.exists('data/labels_map.json'):
        with open('data/labels_map.json', 'r') as f:
            _labels_map = json.load(f)
    return _labels_map


def load_domain_labels_map():
    _domain_labels_map = {}
    if os.path.exists('data/domain_labels_map.json'):
        with open('data/domain_labels_map.json', 'r') as f:
            _domain_labels_map = json.load(f)
    return _domain_labels_map


def extract_language_from_path(path):
    filename = path.split('/')[-1]
    file_type = filename.split('.')[-1]
    if file_type == 'py':
        return 'python'
    elif file_type == 'java':
        return 'java'
    elif file_type == 'c':
        return 'c'
    elif file_type == 'cpp':
        return 'cpp'
    elif file_type == 'php':
        return 'php'
    elif file_type == 'sql':
        return 'sql'
    elif file_type == 'rb':
        return 'ruby'
    elif file_type == 'js':
        return 'javascript'
    elif file_type == 'swift':
        return 'swift'
    elif file_type == 'ts':
        return 'typescript'
    elif file_type == 'kt':
        return 'kotlin'
    elif file_type == 'scala':
        return 'scala'
    elif file_type == 'go':
        return 'go'
    elif file_type == 'rs':
        return 'rust'
    elif file_type == 'dart':
        return 'dart'
    elif file_type == 'groovy':
        return 'groovy'
    elif file_type == 'html':
        return 'html'
    elif file_type == 'xml':
        return 'xml'
    elif file_type == 'sh':
        return 'bash'
    elif file_type == 'pl':
        return 'perl'
    elif file_type == 'r':
        return 'r'
    elif file_type == 'lua':
        return 'lua'
    elif file_type == 'hs':
        return 'haskell'
    elif file_type == 'clj':
        return 'clojure'


def create_dataset_file(data_dir, file_name="twinz_code_dup", file_type="parquet", types_to_collect=None):
    if not types_to_collect or not isinstance(types_to_collect, list):
        types_to_collect = ["csv"]
    if file_type != "parquet":
        raise ValueError(f"{file_type} is currently not supported. Please use parquet.")
    data_dir = os.path.join(SCRIPT_DIR, data_dir)
    os.makedirs(data_dir, exist_ok=True)
    data_files = glob.glob(f"{data_dir}/*.{'|'.join(types_to_collect)}")
    data = pd.concat([pd.read_csv(file) for file in data_files], ignore_index=True)
    # save as parquet dataset
    data.to_parquet(f"{data_dir}/{file_name}.{file_type}", index=False)

    labels = data['label'].unique()
    domain_labels = data['domain_label'].unique()
    labels = list(labels)
    domain_labels = list(domain_labels)
    features = datasets.Features({
        'code': datasets.Value('string', id=None),
        'label': datasets.ClassLabel(names=labels, num_classes=len(labels), id=None),
        'domain_label': datasets.ClassLabel(names=domain_labels, num_classes=len(domain_labels), id=None),
        'index': datasets.Value('string')
    })

    dataset = datasets.Dataset.from_pandas(data, features=features)
    dataset.save_to_disk(f"{data_dir}/{file_name}")
    dataset.push_to_hub("Guychuk/code-duplicates-across-languages", token=os.environ["HF_TOKEN"])


def split_dataset(dataset_id, test_size=0.1):
    dataset = load_dataset(dataset_id, split='train')
    dataset = dataset.train_test_split(test_size=test_size, train_size=1 - test_size)

    # save to disk
    dataset['train'].save_to_disk(f"{SCRIPT_DIR}/data/train")
    dataset['test'].save_to_disk(f"{SCRIPT_DIR}/data/test")

    # push to hub
    dataset.push_to_hub("Guychuk/code-duplicates-across-languages", token=os.environ['HF_TOKEN'])


def push_ready_dataset_to_hub(dataset_id, dataset_dir):
    dataset = datasets.load_from_disk(dataset_dir)
    dataset.push_to_hub(dataset_id, token="")
    return dataset


push_ready_dataset_to_hub("Guychuk/code-duplicates-across-languages", "data/twinz_code_dup")
