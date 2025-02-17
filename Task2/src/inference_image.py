import os
import logging
from dotenv import load_dotenv
from argparse import ArgumentParser
import joblib
import torch
import torch.nn as nn
import torchvision
from torchvision.models import EfficientNet_B3_Weights
from torchvision.transforms import Compose
from utils import load_image_model

# Load constants from .env file
load_dotenv()
IMAGES_MODEL_PATH = os.getenv('IMAGES_MODEL_PATH')
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL')
LOGGING_FORMAT = os.getenv('LOGGING_FORMAT')
LOGGING_DATE_FORMAT = os.getenv('LOGGING_DATE_FORMAT')


def load_preprocess_image(image_path: str, transform: Compose) -> torch.Tensor:
    """
    Preprocess an image for inference
    
    Args:
        image_path (str): Path to the image
        transform (Compose): Image transformation pipeline
        device (str): Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    image = torchvision.io.read_image(image_path)
    image = transform(image)
    return image.unsqueeze(0)


def predict(model: nn.Module, image_path: str, transform: Compose, device: str) -> int:
    """
    Make predictions on one or multiple images
    
    Args:
        model (nn.Module): Image model to use for inference
        image_path (str): Path to the image
        transform (Compose): Image transformation pipeline
        device (str): Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        int: Predicted class label
    """
    model.eval()
    with torch.no_grad():
        image = load_preprocess_image(image_path, transform).to(device)
        output = model(image)
        pred = output.argmax(dim=1).item()
    
    return pred


def main(image_path: str, model_path: str = None, idx_to_class_path: str = None) -> tuple[int, str]:
    """
    Run inference on images
    
    Args:
        image_path (str): Path to image
        model_path (str, optional): Path to the saved model weights
        idx_to_class_path (str, optional): Path to the index to class mapping (pkl file with dict)

    Returns:
        (int, str): Predicted class label and class name
    """
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)
    
    # Check if the arguments are provided, otherwise use the environment variables
    model_path = model_path if model_path else os.path.join(IMAGES_MODEL_PATH, 'image_model.pth')
    idx_to_class_path = idx_to_class_path if idx_to_class_path else os.path.join(IMAGES_MODEL_PATH, 'idx_to_class.pkl')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info('Loading index to class mapping')
    idx_to_class = joblib.load(idx_to_class_path)

    logging.info('Loading model')
    model = load_image_model(device, num_classes=len(idx_to_class), model_path=model_path)
    
    transform = torchvision.transforms.Compose([
        EfficientNet_B3_Weights.DEFAULT.transforms(),
    ])
    pred = predict(model, image_path, transform, device)
    logging.info(f'Image: {image_path}, Prediction: {pred} ({idx_to_class[pred]})')

    return pred, idx_to_class[pred]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the saved model weights')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image (required)')
    parser.add_argument('--idx_to_class_path', type=str, help='Path to the index to class mapping (pkl file with dict)')

    args = parser.parse_args()

    main(args.image_path, args.model_path, args.idx_to_class_path)