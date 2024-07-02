import logging
import os
import torch
from datasets import load_dataset, ClassLabel
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import random
from tqdm import tqdm
from CodeDupModel.utils import create_dataset_file
from config import Config
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer
from model_logic import Model
from dataset import TextDataset

# Initialize logging
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, tokenizer, dataset, device, config):
    train_dataset = TextDataset(tokenizer, dataset, config['max_seq_length'])

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config['batch_size'])

    logger.info(f"Train dataset length: {len(train_dataset)}")

    total_steps = len(train_dataloader) * config['num_train_epochs']
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], eps=config['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=total_steps)
    logger.info(f"Total steps: {total_steps}")

    model.to(device)
    model.train()

    global_step = 0
    logger.info("Start training")

    for epoch in range(config['num_train_epochs']):
        logger.info(f"Epoch: {epoch + 1}")
        for step, batch in tqdm(enumerate(train_dataloader), desc="Training", total=len(train_dataloader)):
            logger.info(f"Step: {step}")
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'positive_input_ids': batch[1], 'negative_input_ids': batch[2], 'labels': batch[3], 'negative_labels': batch[4], 'domain_labels': batch[5], 'alpha': config['alpha']}
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if step % 50 == 0:
                logger.info(f"Epoch: {epoch + 1}, Step: {step}, Loss: {loss.item()}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mps_available = torch.backends.mps.is_available()
    if mps_available:
        torch.backends.mps.enable_mps()
        logger.info("MPS is enabled")
        device = torch.device("mps")
    logger.info(f"Using device: {device}")

    config = Config()

    set_seed(42)

    model_config = RobertaConfig.from_pretrained(config['model_name'])
    tokenizer = RobertaTokenizer.from_pretrained(config['model_name'])
    model = RobertaModel.from_pretrained(config['model_name'], config=model_config)
    # Training
    dataset = load_dataset('Guychuk/code-duplicates-across-languages', split='train')

    domain_labels = dataset.unique('domain_label')
    languages = list(domain_labels)

    model = Model(model, config, tokenizer, languages)

    logger.info(f"Training/evaluation parameters: {config}")

    train(model, tokenizer, dataset, device, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)
    main()
