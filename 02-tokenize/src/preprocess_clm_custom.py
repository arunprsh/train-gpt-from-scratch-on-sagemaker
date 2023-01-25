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
CHUNK_SIZE = 128
N_GPUS = 1

# Re-create GPT2 tokenizer using the saved custom vocabulary from the previous job
logger.info(f'Re-creating GPT2 tokenizer using custom vocabulary from [{LOCAL_INPUT_PATH}/vocab/]')
tokenizer = GPT2TokenizerFast.from_pretrained(f'{LOCAL_INPUT_PATH}', pad_token='<|endoftext|>')
tokenizer.model_max_length = MAX_LENGTH
logger.info(f'Tokenizer: {tokenizer}')

# Read dataset and collate to create mini batches for Masked Language Model (MLM) training
logger.info('Reading and collating input data to create mini batches for Causal Language Model (CLM) training')
dataset = load_dataset('text', data_files=f'{LOCAL_INPUT_PATH}/data/covid_articles.txt', split='train', cache_dir='/tmp/cache')
logger.info(f'Dataset: {dataset}')

# Split dataset into train and validation splits 
logger.info('Splitting dataset into train and validation splits')
train_test_splits = dataset.train_test_split(shuffle=True, seed=123, test_size=0.1)
data_splits = DatasetDict({'train': train_test_splits['train'], 
                           'validation': train_test_splits['test']})
logger.info(f'Data splits: {data_splits}')

