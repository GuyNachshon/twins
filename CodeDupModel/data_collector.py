import csv
import os.path
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import login
import tqdm
from datasets import load_dataset
import json
from sqids import Sqids
from openai import OpenAI
import tiktoken

from CodeDupModel.utils import load_labels_map, load_domain_labels_map, extract_language_from_path

SUPPORTED_LANGUAGES = [
    'python', 'c', 'cpp', 'php', 'sql', 'ruby', 'javascript', 'java', 'c', 'swift',
    'typescript', 'kotlin', 'scala', 'go', 'rust', 'dart', 'groovy',
    'bash', 'perl', 'r', 'lua', 'haskell', 'clojure'
]

SQIDS = Sqids(alphabet="abcdefghijklmnopqrstuvwxyz0123456789")

labels_map = load_labels_map()
domain_labels_map = load_domain_labels_map()
labels_map_mutex = threading.Lock()
domain_labels_map_mutex = threading.Lock()

total_price_lock = threading.Lock()
stop_processing = False

openai = OpenAI(api_key="")
tokens_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

token_price = 5 / 1000000
total_price = 0


def write_to_label_map():
    with labels_map_mutex:
        with open("data/labels_map.json", "w+") as f:
            json.dump(labels_map, f)


def write_to_domain_label_map():
    with domain_labels_map_mutex:
        with open("data/domain_labels_map.json", "w+") as f:
            json.dump(domain_labels_map, f)


class DataItem:
    """
    code: The source code snippet.
    label: The label for the code (used for identifying similar snippets).
    domain_label: The label indicating the programming language or domain.
    index: A unique identifier for the snippet.
    """

    def __init__(self, code, label, domain_label):
        self.code = code
        self.domain_label = domain_label
        self.label = label
        self.normalize_domain_label()
        self.normalize_code()
        self.label_num = self.map_label()
        self.domain_label_num = self.map_domain_label()

        self.index = SQIDS.encode([self.label_num, self.domain_label_num])

    def __str__(self):
        return f"DataItem(code: {self.code}, label: {self.label}, domain_label: {self.domain_label}, index: {self.index})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.code == other.code and self.label == other.label and self.domain_label == other.domain_label and self.index == other.index

    def __hash__(self):
        return hash((self.code, self.label, self.domain_label, self.index))

    def map_label(self):
        if self.label not in labels_map:
            labels_map[self.label] = len(labels_map) + 1
        write_to_label_map()
        return labels_map[self.label]

    def map_domain_label(self):
        if self.domain_label not in domain_labels_map:
            domain_labels_map[self.domain_label] = len(domain_labels_map) + 1
        write_to_domain_label_map()
        return domain_labels_map[self.domain_label]

    def normalize_code(self):
        self.code = self.code.strip()
        c_style_pattern = r'//.*|/\*[\s\S]*?\*/'
        if self.domain_label in ['python', 'c', 'cpp', 'php', 'sql', 'ruby']:
            pattern = r"(#.*)|(\"{3}[\s\S]*?\"{3})|(\"[\s\S]*?\")|(\/\/.*)|(\/\*[\s\S]*?\*\/)"
            self.code = re.sub(pattern, '', self.code)
        elif self.domain_label in ['javascript', 'java', 'c', 'swift', 'typescript', 'kotlin', 'scala', 'go', 'rust', 'dart', 'groovy']:
            self.code = re.sub(c_style_pattern, '', self.code, flags=re.DOTALL)
        elif self.domain_label in ['html', 'xml']:
            self.code = re.sub(r'<!--.*?-->', '', self.code, flags=re.DOTALL)
        elif self.domain_label in ['bash', 'perl', 'r']:
            self.code = re.sub(r'#.*', '', self.code)
        elif self.domain_label == 'lua':
            self.code = re.sub(r'--.*|(?s)--\[\[.*?]]', '', self.code)
        elif self.domain_label == 'haskell':
            self.code = re.sub(r'--.*|\{-[\s\S]*?-}', '', self.code)
        elif self.domain_label == 'clojure':
            self.code = re.sub(r';.*', '', self.code)
        self.code = self.code.strip()
        self.code = self.code.encode('ascii', 'ignore').decode('utf-8')

    def normalize_domain_label(self):
        self.domain_label = self.domain_label.lower()


def collect_rosetta():
    dataset = load_dataset("cakiki/rosetta-code", split="train")
    processed_data = []
    for snippet in tqdm.tqdm(dataset, desc="Processing Rosetta Code snippets", total=len(dataset)):
        task_name = snippet['task_name']
        code = snippet['code']
        language_name = snippet['language_name']
        if language_name.lower() not in SUPPORTED_LANGUAGES:
            continue
        processed_data.append(DataItem(code, task_name, language_name))
    return processed_data


def translate_code(code, source_lang, target_lang):
    prompt = f"Translate {source_lang} code to {target_lang}:\n{code}"
    model_name = "gpt-4o"
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    res = response.choices[0].message.content
    return res


def process_snippet(snippet, processed_data, stop_event):
    global total_price

    code = snippet['content']
    if not code or stop_event.is_set():
        return
    repo_name = snippet['repo_name']
    path = snippet['path']
    label = f"{repo_name}/{path}"
    language = extract_language_from_path(path)

    if language not in SUPPORTED_LANGUAGES:
        return

    data_item = DataItem(code, label, language)

    with total_price_lock:
        code_tokens = tokens_encoding.encode(data_item.code)
        if len(code_tokens) > 16384:
            code_tokens = code_tokens[:16384]
        data_item.code = tokens_encoding.decode(code_tokens)
        price_increase = len(code_tokens) * token_price
        total_price += price_increase
        if total_price > 200:
            stop_event.set()

    if stop_event.is_set():
        return

    processed_data.append(data_item)

    for target_lang in random.sample(SUPPORTED_LANGUAGES, 5):
        if target_lang == language or stop_event.is_set():
            continue

        translated_code = translate_code(data_item.code, language, target_lang)

        with total_price_lock:
            code_tokens = tokens_encoding.encode(translated_code)
            price_increase = len(code_tokens) * token_price
            total_price += price_increase
            if total_price > 200:
                stop_event.set()
                break

        processed_data.append(DataItem(translated_code, label, target_lang))


def collect_github():
    login(token="")
    dataset = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=True)
    processed_data = []
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        pbar = tqdm.tqdm(desc="Processing GitHub snippets", total=477028)
        pbar_future = tqdm.tqdm(desc="updating futures", total=477028)
        for snippet in dataset:
            future = executor.submit(process_snippet, snippet, processed_data, stop_event)
            pbar.update(1)
            futures.append(future)
            pbar_future.update(1)
        try:
            for future in as_completed(futures):
                if stop_event.is_set():
                    break
        except KeyboardInterrupt:
            stop_event.set()
            executor.shutdown(wait=False)
            raise

    if not stop_event.is_set():
        print("Finished processing without reaching the price limit.")
    else:
        print("Stopped processing due to reaching the price limit.")

    # Save processed data to a file
    with open("data/github_temp.csv", "w+") as f:
        f.write("index,domain_label,label,code\n")
        for item in processed_data:
            f.write(f"{item.index},{item.domain_label},{item.label},{item.code}\n")

    return processed_data


def main():
    if os.path.exists("data/rosetta.csv"):
        with open("data/rosetta.csv", "r") as f:
            rosetta = csv.DictReader(f)
    else:
        rosetta = collect_rosetta()
        rosetta_path = "data/rosetta.csv"
        with open(rosetta_path, "w+") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "domain_label", "label", "code"])
            writer.writeheader()
            for item in rosetta:
                writer.writerow({"index": item.index, "domain_label": item.domain_label, "label": item.label, "code": item.code})

    if os.path.exists("data/github.csv"):
        with open("data/github.csv", "r") as f:
            github = csv.DictReader(f)
    else:
        print("collecting github")
        github = collect_github()
        github_path = "data/github.csv"
        with open(github_path, "w+") as f:
            f.write("index,domain_label,label,code\n")
            for item in github:
                f.write(f"{item.index},{item.domain_label},{item.label},{item.code}\n")


if __name__ == '__main__':
    main()
