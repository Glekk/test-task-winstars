# Task1

## Contents
- [Solution Overview](#solution-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Solution Overview
This solution is aimed to solve MNIST image classification task. I have implemented three different models: Random Forest (scikit-learn), Neural Network and Convolutional Neural Network (both using PyTorch). The models are trained using the training part of MNIST and then tested on the test part. The performance of the models is evaluated using the accuracy metric. All models are saved in the models folder and could be used for inference. You can find more information in the demo notebook in the `notebooks` folder.

## Project Structure 
- `data` folder - contains the preprocessed data
- `models` folder - contains the saved models in .pkl format
- `notebooks` folder - contains the notebooks with demonstration of the solution
- `src` folder - contains the source code
    - `classifiers` folder - contains model classes
    - `train.py` - script for training the models
    - `inference.py` - script for inference
    - `utils.py` - utility functions
- `.env` (included as .env.example) - environment variables file
- `requirements.txt` - requirements file

## Prerequisites
- Python 3.10 (tested on this version)
- pip
- Jupiter Notebook\VS Code with extensions (to run the notebooks)

### Installation
1. Clone the repository
```
git clone https://github.com/Glekk/test-task-winstars.git
```
2. Navigate to the desired directory
```
cd test-task-winstars/Task1
```
3. Install the requirements
```
pip install -r requirements.txt
```

## Usage
- Before all you need check and if needed change the environment variables in .env file. There is `.env.example` as an example.
- The data is saved in the data folder (but it automatically checks and redownloads if something wrong). Also there are saved models in the models folder.
- To run the training you need to run train.py in src folder (you can run it from different folder, but then you may need to change paths in .env):
```
cd src
python train.py --alg <algorithm to train the model>
```
--alg accepts three values: 'rf' for Random Forest, 'nn' for Neural Network and 'cnn' for Convolutional Neural Network. Default value is 'rf'.
- Same for inference.py (it validates the model on the test part of MNIST):
```
cd src
python inference.py --alg <algorithm to use for inference>
```
- To run the notebook you just need to open it in appropriate environment (Jupiter Notebook\VS Code) and run all cells.
