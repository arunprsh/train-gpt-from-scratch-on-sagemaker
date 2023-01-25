from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
from pathlib import Path
import transformers 
import tokenizers
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
logger.info(f'[Using Tokenizers: {tokenizers.__version__}]')

# Essentials
# LOCAL_INPUT_PATH is mapped to S3 input location for covid news articles 
LOCAL_INPUT_PATH = '/opt/ml/processing/input' 
# LOCAL_OUTPUT_PATH is mapped to S3 output location where we want to save the custom vocabulary after training the tokenizer
LOCAL_OUTPUT_PATH = '/opt/ml/processing/output'

# Read input files from local input path 
logger.info(f'Reading input files from [{LOCAL_INPUT_PATH}/]')
paths = [str(x) for x in Path(LOCAL_INPUT_PATH).glob('*.txt')]

# Derive default vocab size and model max length values from default GPT2 tokenizer 
default_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
vocab_size = default_tokenizer.vocab_size
model_max_length = default_tokenizer.model_max_length

# Train custom tokenizer
logger.info(f'Train custom ByteLevelBPEPair tokenizer using files in {paths}')
custom_tokenizer = ByteLevelBPETokenizer(lowercase=True)
custom_tokenizer.train(files=paths, 
                vocab_size=vocab_size, 
                min_frequency=1, 
                special_tokens=['<|endoftext|>'])
custom_tokenizer.enable_truncation(max_length=model_max_length)

# Save trained custom tokenizer to local output path
logger.info(f'Saving extracted custom vocabulary to [{LOCAL_OUTPUT_PATH}/]')
custom_tokenizer.save_model(LOCAL_OUTPUT_PATH)

# Re-create custom tokenizer using vocab from local output path
logger.info(f'Re-create custom GPT2Tokenizer using the extracted vocabulary in {LOCAL_OUTPUT_PATH}')
custom_tokenizer = GPT2TokenizerFast.from_pretrained(f'{LOCAL_OUTPUT_PATH}', pad_token='<|endoftext|>')
custom_tokenizer.model_max_length = model_max_length

# Evaluate custom tokenizer 
logger.info('Evaluating custom tokenizer')
test_sentence = 'covid virus in usa'
logger.info(f'Test sentence: {test_sentence}')
tokens = custom_tokenizer.encode(test_sentence).tokens
logger.info(f'Encoded sentence: {tokens}')
token_id = custom_tokenizer.token_to_id('covid')
logger.info(f'Token ID for token (covid) = {token_id}')
vocab_size = custom_tokenizer.get_vocab_size()
logger.info(f'Vocabulary size = {vocab_size}')