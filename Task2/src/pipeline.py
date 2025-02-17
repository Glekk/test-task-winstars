import os
import logging
from dotenv import load_dotenv
from argparse import ArgumentParser
from inference_image import main as inference_image
from inference_ner import main as inference_ner
from utils import get_animals_from_ner


# Load constants from .env file
load_dotenv()
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL')
LOGGING_FORMAT = os.getenv('LOGGING_FORMAT')
LOGGING_DATE_FORMAT = os.getenv('LOGGING_DATE_FORMAT')


def main(text: str, image_path: str, image_model_path: str = None, text_model_path: str = None, idx_to_class_path: str = None, printing: bool = False) -> bool:
    '''
    Run the image-text inference pipeline.

    Args:
        text (str): The text to perform NER on.
        image_path (str): Path to image or directory of images
        image_model_path (str, optional): Path to the saved image model weights
        text_model_path (str, optional): Path to the saved text model weights
        idx_to_class_path (str, optional): Path to the index to class mapping (pkl file with dict)
        printing (bool, default=False): Whether to print the NER result.

    Returns:
        bool: True if the animal from the text is found in the image, False otherwise.
    '''
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)

    # Run text inference
    logging.info('Running NER inference on text')
    result = inference_ner(text, text_model_path, printing)
    text_animals = get_animals_from_ner(result)

    # Run image inference
    logging.info('Running image inference')
    _, image_animal = inference_image(image_path, image_model_path, idx_to_class_path)
    image_animal = image_animal.split('_')
    image_animal = ' '.join(image_animal)
    
    # Check if the animal from the text is found in the image
    found = False
    for animal in text_animals:
        if animal.lower() in image_animal:
            found = True
            break

    if printing:
        logging.info(f'Animal from text: {text_animals}')
        logging.info(f'Animal from image: {image_animal}')
        logging.info(f'Is the animal from the text found in the image?: {found}')

    return found

        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help='The text to perform NER on (required)')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image (required)')
    parser.add_argument('--image_model_path', type=str, help='Path to the saved image model weights')
    parser.add_argument('--text_model_path', type=str, help='Path to the saved text model weights')
    parser.add_argument('--idx_to_class_path', type=str, help='Path to the index to class mapping (pkl file with dict)')
    parser.add_argument('--no_print', action='store_false', help='Do not print the NER result.')

    args = parser.parse_args()
    text = args.text
    image_path = args.image_path
    image_model_path = args.image_model_path
    text_model_path = args.text_model_path
    idx_to_class_path = args.idx_to_class_path
    no_print = args.no_print

    main(text, image_path, image_model_path, text_model_path, idx_to_class_path, no_print)
