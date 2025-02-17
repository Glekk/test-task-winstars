import logging
import joblib
from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import train_epoch


class MnistClassifierInterface(ABC):
    '''Interface for MNIST classifiers'''
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Train the model with the given data

        Args:
            X (np.ndarray): an input data of shape (n_samples, n_features)
            y (np.ndarray): labels of the data with shape (n_samples,)
        '''
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Make predictions on the given data

        Args:
            X (np.ndarray): an input data of shape (n_samples, n_features)

        Returns:
            predictions (np.ndarray): an array of shape (n_samples,)
        '''
        pass


class RandomForestMnistClassifier(MnistClassifierInterface):
    '''
    Random Forest classifier for MNIST dataset
    
    Args:
        kwargs (dict, default={'random_state': 42}): parameters for the RandomForestClassifier
    '''
    def __init__(self, kwargs: dict={'random_state': 42}):
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_reshaped = X.reshape(X.shape[0], -1)
        self.model.fit(X_reshaped, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_reshaped = X.reshape(X.shape[0], -1)
        return self.model.predict(X_reshaped)
    

class NeuralNetworkMnistClassifier(MnistClassifierInterface):
    '''
    Neural Network classifier for MNIST dataset
    
    Args:
        device (str, default=autodetect): device to use for training the model
        criterion (torch.nn.Module, default=nn.CrossEntropyLoss()): loss function to use
        optimizer (torch.optim.Optimizer, default=optim.Adam): optimizer to use
        lr (float, default=0.001): learning rate for the optimizer
        epochs (int, default=10): number of epochs to train the model
        classes (int, default=10): number of classes in the dataset
    '''
    def __init__(self, device: str='cuda' if torch.cuda.is_available() else 'cpu', 
                 criterion: nn.Module=nn.CrossEntropyLoss(), 
                 optimizer: optim.Optimizer=optim.Adam, lr: float=1e-3, epochs: int=10, classes: int=10):
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, classes)
        ).to(device)

        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_reshaped = X.reshape(X.shape[0], -1)

        X_tensor = torch.tensor(X_reshaped, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        for epoch in range(self.epochs):
            loss, acc = train_epoch(self.model, train_loader, self.criterion, self.optimizer, self.device)
            logging.info(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss}, Accuracy: {acc}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_reshaped = X.reshape(X.shape[0], -1)
        X_tensor = torch.tensor(X_reshaped, dtype=torch.float32).to(self.device)
        y_pred = self.model(X_tensor)
        return torch.argmax(y_pred, dim=1).cpu().numpy()


class CNNMnistClassifier(MnistClassifierInterface):
    '''
    Convolutional Neural Network classifier for MNIST dataset

    Args:
        device (str, default=autodetect): device to use for training the model
        criterion (torch.nn.Module, default=nn.CrossEntropyLoss()): loss function to use
        optimizer (torch.optim.Optimizer, default=optim.Adam): optimizer to use
        lr (float, default=0.001): learning rate for the optimizer
        epochs (int, default=10): number of epochs to train the model
        classes (int, default=10): number of classes in the dataset
    '''
    def __init__(self, device: str='cuda' if torch.cuda.is_available() else 'cpu', 
                 criterion: nn.Module=nn.CrossEntropyLoss(), 
                 optimizer: optim.Optimizer=optim.Adam, lr: float=1e-3, epochs: int=10, classes: int=10):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, classes)
        ).to(device)

        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_tensor = torch.tensor(X.reshape(-1, 1, 28, 28), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        for epoch in range(self.epochs):
            loss, acc = train_epoch(self.model, train_loader, self.criterion, self.optimizer, self.device)
            logging.info(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss}, Accuracy: {acc}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.tensor(X.reshape(-1, 1, 28, 28), dtype=torch.float32).to(self.device)
        y_pred = self.model(X_tensor)
        return torch.argmax(y_pred, dim=1).cpu().numpy()


class MnistClassifier:
    '''
    Wrapper class for all MNIST classifiers

    Args:
        algorithm (str): name of the algorithm to use (rf, nn or cnn)
        kwargs (dict, default={}): parameters for the classifier
    '''
    def __init__(self, algorithm: str):
        self.__options = {
            'rf': RandomForestMnistClassifier,
            'nn': NeuralNetworkMnistClassifier,
            'cnn': CNNMnistClassifier
        }
        
        self.model = self.__options[algorithm]() 

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Train selected model with the given data

        Args:
            X (np.ndarray): an input data of shape (n_samples, n_features)
            y (np.ndarray): labels of the data with shape (n_samples,)
        '''
        self.model.train(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Make predictions on the given data

        Args:
            X (np.ndarray): an input data of shape (n_samples, n_features)

        Returns:
            predictions (np.ndarray): an array of shape (n_samples,)
        '''
        return self.model.predict(X)
    

def save_model(clf: MnistClassifier, model_path: str) -> None:
    '''
    Save the trained model to the specified path

    Args:
        clf (MnistClassifier): trained model
        model_path (str): path to save the model
    '''
    with open(model_path, 'wb') as f:
        joblib.dump(clf, f, compress=3)


def load_model(model_path: str) -> MnistClassifier:
    '''
    Load the trained model from the specified path
    
    Args:
        model_path (str): path to load the model from
        
    Returns:
        clf (MnistClassifier): trained model
    '''
    with open(model_path, 'rb') as f:
        clf = joblib.load(f)
    
    return clf
