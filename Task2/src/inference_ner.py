import os
import logging
from dotenv import load_dotenv
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from utils import print_ner_result

# Load constants from .env file
load_dotenv()
TEXT_MODEL_PATH = os.getenv('TEXT_MODEL_PATH')
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL')
LOGGING_FORMAT = os.getenv('LOGGING_FORMAT')
LOGGING_DATE_FORMAT = os.getenv('LOGGING_DATE_FORMAT')


def main(text: str, model_path: str = None, printing: bool = True) -> list:
    '''
    Perform NER on the given text.

    Args:
        model_path (str, optional): The path to the saved model weights.
        text (str): The text to perform NER on.
        printing (bool, default=True): Whether to print the NER result.

    Returns:
        list: The result of the NER pipeline.
    '''
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)

    model_path = model_path if model_path else TEXT_MODEL_PATH

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    logging.info('Model loaded')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logging.info('Tokenizer loaded')

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    logging.info('NER pipeline created')

    result = ner_pipeline(text)
    logging.info(f"Result: {result}")

    if printing:
        print_ner_result(result)

    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the saved model weights')
    parser.add_argument('--text', type=str, required=True, help='The text to perform NER on (required)')
    parser.add_argument('--no_print', action='store_false', help='Do not print the NER result')
    args = parser.parse_args()

    model_path = args.model_path
    text = args.text
    no_print = args.no_print
    
    main(text, model_path, no_print)