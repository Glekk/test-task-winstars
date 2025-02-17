import os
import logging
from dotenv import load_dotenv
from argparse import ArgumentParser
import joblib
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.utils.data import DataLoader, random_split
from utils import set_seed, get_accuracy, load_image_model, evaluate_image, check_folders_existence


# Load constants from .env file
load_dotenv()
SEED = int(os.getenv('SEED'))
IMAGES_DATASET_RAW_PATH = str(os.getenv('IMAGES_DATASET_RAW_PATH'))
IMAGES_MODEL_PATH = str(os.getenv('IMAGES_MODEL_PATH'))
LOGGING_LEVEL = str(os.getenv('LOGGING_LEVEL'))
LOGGING_FORMAT = str(os.getenv('LOGGING_FORMAT'))
LOGGING_DATE_FORMAT = str(os.getenv('LOGGING_DATE_FORMAT'))
LEARNING_RATE_IMAGE_MODEL = float(os.getenv('LEARNING_RATE_IMAGE_MODEL'))
EPOCHS_IMAGE_MODEL = int(os.getenv('EPOCHS_IMAGE_MODEL'))
BATCH_SIZE_IMAGE_MODEL = int(os.getenv('BATCH_SIZE_IMAGE_MODEL'))
NUM_WORKERS = int(os.getenv('NUM_WORKERS'))


def get_trainval_dataloader(path: str, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    '''
    Load dataset from path and return a DataLoader object
    
    Args:
        path (str): path to the dataset
        batch_size (int): batch size for the DataLoader object
        num_workers (int): number of workers for the DataLoader object

    Returns:
        (DataLoader, DataLoader): a tuple containing the train and validation DataLoader objects
    '''
    dataset = ImageFolder(path, transform=Compose([
        EfficientNet_B3_Weights.DEFAULT.transforms(),
        RandomHorizontalFlip(),
        RandomVerticalFlip()
    ]))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader
    

def train_image_model(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer,
                      train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int, device: str) -> dict[str, list[float]]:
    '''
    Train an image model
    
    Args:
        model (nn.Module): image model to train
        criterion (nn.Module): loss function
        optimizer (optim.Optimizer): optimizer
        train_dataloader (DataLoader): DataLoader object containing the training dataset
        val_dataloader (DataLoader): DataLoader object containing the validation dataset
        
    Returns:
        dict(str, list[float]): history of the training process
    '''
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    for epoch in range(epochs):
        loop = tqdm(train_dataloader)
        model.train()
        running_train_loss = 0.0
        running_train_acc = 0.0
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            acc = get_accuracy(preds, labels).item()
            running_train_acc += acc

            loop.set_description(f'Epoch {epoch + 1}/{EPOCHS_IMAGE_MODEL}')
            loop.set_postfix(loss=loss.item(), accuracy=acc)

        running_train_loss /= len(train_dataloader)
        running_train_acc /= len(train_dataloader)
        history['train_loss'].append(running_train_loss)
        history['train_acc'].append(running_train_acc)

        running_val_loss, running_val_acc = evaluate_image(model, criterion, val_dataloader, device)

        history['val_loss'].append(running_val_loss)
        history['val_acc'].append(running_val_acc)

        logging.info(f'Epoch {epoch + 1}/{EPOCHS_IMAGE_MODEL}, Train Loss: {running_train_loss}, Train Accuracy: {running_train_acc}, Validation Loss: {running_val_loss}, Validation Accuracy: {running_val_acc}')

    return history


def main(seed:int = None, path: str= None, batch_size: int= None, 
         learning_rate: float= None, epochs: int= None, model_save_path: str= None, num_workers: int= None) -> None:
    '''
    Train an image model

    Args:
        seed (int, optional): seed for reproducibility
        path (str, optional): path to the dataset
        batch_size (int, optional): batch size for the DataLoader object
        learning_rate (float, optional): learning rate for the optimizer
        epochs (int, optional): number of epochs to train the model
        model_save_path (str, optional): path to save the model
        num_workers (int, optional): number of workers for the DataLoader object

    '''
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)
    
    # Check if the arguments are provided, otherwise use the environment variables
    seed = seed if seed else SEED
    path = path if path else IMAGES_DATASET_RAW_PATH
    batch_size = batch_size if batch_size else BATCH_SIZE_IMAGE_MODEL
    learning_rate = learning_rate if learning_rate else LEARNING_RATE_IMAGE_MODEL
    epochs = epochs if epochs else EPOCHS_IMAGE_MODEL
    model_save_path = model_save_path if model_save_path else os.path.join(IMAGES_MODEL_PATH, 'image_model.pth')
    num_workers = num_workers if num_workers else NUM_WORKERS

    # Set seed for reproducibility
    set_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Check if the folder for the model exists, otherwise create it
    check_folders_existence([model_save_path])

    logging.info('Loading dataset')
    # Load dataset
    train_dataloader, val_dataloader = get_trainval_dataloader(path, batch_size, num_workers)
    class_to_idx = train_dataloader.dataset.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = load_image_model(device, len(idx_to_class), EfficientNet_B3_Weights.DEFAULT)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logging.info('Training image model')
    history = train_image_model(model, criterion, optimizer, train_dataloader, val_dataloader, epochs, device)

    logging.info('Saving image model')
    torch.save(model.state_dict(), model_save_path)

    logging.info('Saving image model metrics')
    joblib.dump(history, os.path.join(os.path.dirname(model_save_path), 'image_model_metrics.pkl'))

    logging.info('Saving class to index mapping')
    joblib.dump(idx_to_class, os.path.join(os.path.dirname(model_save_path), 'idx_to_class.pkl'))
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, help='Seed for reproducibility')
    parser.add_argument('--path', type=str, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, help='Batch size for the DataLoader object')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train the model')
    parser.add_argument('--model_save_path', type=str, help='Path to save the model')
    parser.add_argument('--num_workers', type=int, help='Number of workers for the DataLoader object')

    args = parser.parse_args()

    seed = args.seed
    path = args.path
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    model_save_path = args.model_save_path
    num_workers = args.num_workers
    
    main(seed, path, batch_size, learning_rate, epochs, model_save_path, num_workers)
