import os
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    '''
    Calculate the accuracy of the model
    
    Args:
        preds (torch.Tensor): model predictions
        labels (torch.Tensor): true labels
        
    Returns:
        float: accuracy of the model
    '''
    if preds.shape != labels.shape:
        preds = preds.argmax(dim=1)

    return (preds == labels).float().mean()


def load_image_model(device: str, num_classes: int, weights: EfficientNet_B3_Weights = None, model_path: str = None) -> nn.Module:
    '''
    Load the trained model
    
    Args:
        device (str): Device to run inference on ('cuda' or 'cpu')
        num_classes (int): Number of classes in the dataset
        weights (EfficientNet_B3_Weights, default=None): Weights to use for the model
        model_path (str, default=None): Path to the model (if specified, will load model weights from this path)

    Returns:
        nn.Module: Loaded model
    '''
    model = efficientnet_b3(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)

    return model


def evaluate_image(model: nn.Module, criterion: nn.Module, 
                   dataloader: DataLoader, device: str, return_preds: bool=False) -> tuple[float, float]|tuple[float, float, list, list]:
    '''
    Evaluate the image model

    Args:
        model (nn.Module): image model to evaluate
        criterion (nn.Module): loss function
        dataloader (DataLoader): DataLoader object containing the dataset to evaluate
        device (str): device to use for the evaluation

    Returns:
        (float, float) or (float, float, list, list): if return_preds is False, returns validation loss and accuracy, otherwise returns validation loss, accuracy, predictions, and labels

    '''
    loop = tqdm(dataloader)
    model.eval()
    running_val_loss = 0.0
    running_val_acc = 0.0
    if return_preds:
        all_preds = []
        all_labels = []
        
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            running_val_loss += loss.item()
            acc = get_accuracy(preds, labels).item()
            running_val_acc += acc
            loop.set_description('Validation')
            loop.set_postfix(loss=loss.item(), accuracy=acc)

            if return_preds:
                all_preds.append(preds.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

    running_val_loss /= len(dataloader)
    running_val_acc /= len(dataloader)

    if return_preds:
        return running_val_loss, running_val_acc, all_preds, all_labels

    return running_val_loss, running_val_acc


def check_folders_existence(folders: list[str]) -> None:
    '''
    Check if the folders exist, otherwise create them
    
    Args:
        folders (list[str]): list of folders to check
    '''
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def print_ner_result(result: list) -> None:
    '''
    Print the result of the NER pipeline with formatting.

    Args:
        result (list): The result of the NER pipeline.
    '''
    for entity in result:
        print(f"Word: {entity['word']}, Start: {entity['start']}, End: {entity['end']}")
        print(f"Entity: {entity['entity']}, Score: {entity['score']}")
        print()


def get_animals_from_ner(result: dict) -> list:
    '''
    Function to extract animals from text based on NER model output, because tokes are split by '##' we need to combine them back

    Args:
        result (dict): The NER model output

    Returns:
        list: List of animals extracted from the text
    '''
    animals = []
    for word in result:
        if word['entity'] == 'B-ANIMAL':
            animals.append(word['word'])
        elif word['entity'] == 'I-ANIMAL':
            if '##' in word['word']:
                animal = word['word'].replace('##', '')
                animals[-1] = animals[-1] + '' + animal
            else:
                animal = word['word']
                animals[-1] = animals[-1] + ' ' + animal
            
    return animals