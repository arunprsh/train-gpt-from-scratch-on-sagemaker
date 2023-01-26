from transformers import GPT2TokenizerFast
from transformers import GPT2Config
from datasets import load_dataset
from datasets import DatasetDict
from pathlib import Path
import transformers 
import datasets
import logging
import sys
import os


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.getLevelName('INFO'), 
                    handlers=[logging.StreamHandler(sys.stdout)], 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Log versions of dependencies
logger.info(f'[Using Transformers: {transformers.__version__}]')
logger.info(f'[Using Datasets: {datasets.__version__}]')

# Essentials
# LOCAL_INPUT_PATH is mapped to S3 input location for covid news articles 
LOCAL_INPUT_PATH = '/opt/ml/processing/input' 
# LOCAL_OUTPUT_PATH is mapped to S3 output location where we want to save the processed input data (COVID articles)
LOCAL_OUTPUT_PATH = '/opt/ml/processing/output'
MAX_LENGTH = 512
N_GPUS = 1

# Use default GPT2 tokenizer 
logger.info(f'Use default GPT2 tokenizer')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', pad_token='<|endoftext|>')
tokenizer.model_max_length = MAX_LENGTH
logger.info(f'Tokenizer: {tokenizer}')

# Read dataset and collate to create mini batches for Causal Language Model (CLM) training
logger.info('Reading and collating input data to create mini batches for Causal Language Model (CLM) training')
dataset = load_dataset('text', data_files=f'{LOCAL_INPUT_PATH}/data/covid_articles.txt', split='train', cache_dir='/tmp/cache')
logger.info(f'Dataset: {dataset}')

# Split dataset into train and validation splits 
logger.info('Splitting dataset into train and validation splits')
train_test_splits = dataset.train_test_split(shuffle=True, seed=123, test_size=0.1)
data_splits = DatasetDict({'train': train_test_splits['train'], 
                           'validation': train_test_splits['test']})
logger.info(f'Data splits: {data_splits}')


def tokenize(element):
    outputs = tokenizer(element['text'], 
                        truncation=True, 
                        max_length=MAX_LENGTH, 
                        return_overflowing_tokens=True, 
                        return_length=True)
    input_batch = []
    for length, input_ids in zip(outputs['length'], outputs['input_ids']):
        if length == MAX_LENGTH:
            input_batch.append(input_ids)
    return {'input_ids': input_batch}


# Tokenize dataset
logger.info('Tokenizing dataset splits')
num_proc = int(os.cpu_count()/N_GPUS)
logger.info(f'Total number of processes = {num_proc}')
tokenized_datasets = data_splits.map(tokenize, 
                                     batched=True, 
                                     num_proc=num_proc,
                                     remove_columns=data_splits['train'].column_names)
logger.info(f'Tokenized datasets: {tokenized_datasets}')

# Save tokenized datasets to local disk (EBS volume)
logger.info(f'Saving tokenized datasets to local disk {LOCAL_OUTPUT_PATH}')
tokenized_datasets.save_to_disk(f'{LOCAL_OUTPUT_PATH}')

# Validate if datasets were saved correctly
logger.info('Validating if datasets were saved correctly')
reloaded_dataset = datasets.load_from_disk(f'{LOCAL_OUTPUT_PATH}')
logger.info(f'Reloaded dataset: {reloaded_dataset}')
