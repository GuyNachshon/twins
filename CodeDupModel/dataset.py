import json
import torch
from torch.utils.data import Dataset
import random
import logging

logger = logging.getLogger(__name__)


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, label, domain_label):
        self.input_ids = input_ids
        self.label = label
        self.domain_label = domain_label


def convert_examples_to_features(example, tokenizer, block_size):
    code = example['code']
    code_tokens = tokenizer.tokenize(code)[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_ids, int(example['label']), int(example['domain_label']))


class TextDataset(Dataset):
    def __init__(self, tokenizer, dataset, block_size):
        self.examples = []
        for item in dataset:
            if not item.get('code'):
                continue
            self.examples.append(convert_examples_to_features(item, tokenizer, block_size))

        self.label_examples = {}
        for example in self.examples:
            domain_label = example.domain_label
            if domain_label not in self.label_examples:
                self.label_examples[domain_label] = {}
            if example.label not in self.label_examples[domain_label]:
                self.label_examples[domain_label][example.label] = []
            self.label_examples[domain_label][example.label].append(example)
        logger.debug(f"Number of examples: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item_index):
        logger.debug(f"Fetching item {item_index}")
        example = self.examples[item_index]
        domain_label = example.domain_label
        label_examples = self.label_examples[domain_label]

        positive_example = random.choice(label_examples[example.label])
        retry_count = 0
        while positive_example == example:
            positive_example = random.choice(label_examples[example.label])
            retry_count += 1
            if retry_count > 10:  # Just a simple safeguard
                break
        logger.debug(f"Selected positive example after {retry_count} retries")

        negative_labels = list(label_examples.keys())
        negative_labels.remove(example.label)
        negative_label = random.choice(negative_labels)
        negative_example = random.choice(label_examples[negative_label])
        logger.debug(f"Selected negative example with label {negative_label}")

        return (torch.tensor(example.input_ids),
                torch.tensor(positive_example.input_ids),
                torch.tensor(negative_example.input_ids),
                torch.tensor(example.label),
                torch.tensor(negative_example.label),
                torch.tensor(example.domain_label))


class TestTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, block_size))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item_index):
        example = self.examples[item_index]
        return (torch.tensor(example.input_ids),
                torch.tensor(example.label),
                torch.tensor(example.domain_label))
