import os
import logging
from dotenv import load_dotenv
import time
from argparse import ArgumentParser
from utils import get_accuracy, load_mnist_data, set_seed
from clasifiers import MnistClassifier, save_model


# Load constants from .env file
load_dotenv()
SEED = int(os.getenv('SEED'))
LOCAL_DATASET_PATH = os.getenv('LOCAL_DATASET_PATH')
MODEL_LOCAL_PATH = os.getenv('MODEL_LOCAL_PATH')
CONFIG_DIR = os.getenv('CONFIG_DIR')
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL')
LOGGING_FORMAT = os.getenv('LOGGING_FORMAT')
LOGGING_DATE_FORMAT = os.getenv('LOGGING_DATE_FORMAT')


def main(algorithm: str) -> None:
    '''
    Train the model with the given algorithm

    Args:
        algorithm (str): the algorithm to train the model
    '''   
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)

    # Set seed for reproducibility
    set_seed(SEED)

    # Load data
    X_train, X_test, y_train, y_test = load_mnist_data(LOCAL_DATASET_PATH)
    
    #Train model
    logging.info(f'Model: {algorithm} training')
    start = time.perf_counter_ns()
    clf = MnistClassifier(algorithm)
    clf.train(X_train, y_train)
    end = time.perf_counter_ns()

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    logging.info(f'Training Time: {(end-start)/1e9} seconds')
    logging.info(f'Train Accuracy: {get_accuracy(y_train, y_pred_train)}')
    logging.info(f'Test Accuracy: {get_accuracy(y_test, y_pred_test)}')
    logging.info('-'*40)

    # Save model
    save_model(clf, os.path.join(MODEL_LOCAL_PATH, f'{algorithm}.pkl'))
       

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--alg', type=str, default='rf', choices=['rf', 'nn', 'cnn'], help='Algorithm to train the model (rf, nn, cnn), default=rf')

    args = parser.parse_args()
    algorithm = args.alg

    main(algorithm)
