import os
import logging
from dotenv import load_dotenv
from argparse import ArgumentParser
import numpy as np
from utils import load_mnist_data, set_seed, get_accuracy
from clasifiers import load_model


# Load constants from .env file
load_dotenv()
SEED = int(os.getenv('SEED'))
MODEL_LOCAL_PATH = os.getenv('MODEL_LOCAL_PATH')
LOCAL_DATASET_PATH = os.getenv('LOCAL_DATASET_PATH')
CONFIG_DIR = os.getenv('CONFIG_DIR')
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL')
LOGGING_FORMAT = os.getenv('LOGGING_FORMAT')
LOGGING_DATE_FORMAT = os.getenv('LOGGING_DATE_FORMAT')


def main(algorithm: str, metrics: bool=True) -> tuple[np.ndarray, np.ndarray]:
    '''
    Make predictions on mnist with the given algorithm

    Args:
        algorithm (str): the algorithm to make predictions
        metrics (bool, default=True): show metrics

    Returns:
        (np.ndarray, np.ndarray): predictions for train and test data
    '''
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)

    # Set seed for reproducibility
    set_seed(SEED)

    # Load data
    X_train, X_test, y_train, y_test = load_mnist_data(LOCAL_DATASET_PATH)

    # Load model
    logging.info(f'Model: {algorithm} loading')
    clf = load_model(os.path.join(MODEL_LOCAL_PATH, f'{algorithm}.pkl'))
    logging.info(f'Making predictions')
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    if metrics:
        logging.info(f'Train Accuracy: {get_accuracy(y_train, y_pred_train)}')
        logging.info(f'Test Accuracy: {get_accuracy(y_test, y_pred_test)}')
        logging.info('-'*40)

    return y_pred_train, y_pred_test


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--alg', type=str, default='rf', choices=['rf', 'nn', 'cnn'], help='Algorithm to use for inference (rf, nn, cnn), default=rf')

    args = parser.parse_args()
    algorithm = args.alg
    
    main(algorithm)
