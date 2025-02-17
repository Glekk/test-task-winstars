import numpy as np
import torch
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_epoch(model: nn.Module, train_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, device: str) -> tuple[float, float]:
    '''
    Train torch model for one epoch

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): data loader for training data
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer to use
        device (str): device to use for training

    Returns:
        (float, float): average loss and accuracy of the model
    '''
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        running_acc += get_accuracy(y, y_pred)
        running_loss += loss.item()
    
    return running_loss/len(train_loader), running_acc/len(train_loader)


def get_accuracy(y_true: np.ndarray|torch.Tensor, y_pred: np.ndarray|torch.Tensor) -> float:
    '''
    Calculate accuracy of the predictions

    Args:
        y_true (np.ndarray|torch.Tensor): true labels
        y_pred (np.ndarray|torch.Tensor): predicted labels

    Returns:
        accuracy (float): accuracy of the predictions
    '''
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().detach().numpy()

    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    return np.mean(y_true == y_pred)


def load_mnist_data(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Load MNIST dataset

    Args:
        path (str): path to save the dataset

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray): train data, test data, train labels, test labels
    '''
    mnist_train = datasets.MNIST(path, download=True, train=True)
    mnist_test = datasets.MNIST(path, download=True, train=False)

    X_train, y_train = mnist_train.data.numpy()/255.0, mnist_train.targets.numpy()
    X_test, y_test = mnist_test.data.numpy()/255.0, mnist_test.targets.numpy()

    return X_train, X_test, y_train, y_test


def set_seed(seed: int) -> None:
    '''
    Set seed for reproducibility

    Args:
        seed (int): seed to set
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
