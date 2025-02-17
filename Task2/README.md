# Task1

## Contents
- [Solution Overview](#solution-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Solution Overview
This solution is aimed to solve Named Entity Recognition (NER) + Image Classification task.  
Pipeline is the following:
- identification of animal names inside the sentence -> classification of the image of the animal -> check if the image is containing the animal name from the sentence or not.

The solution for NER is based on the BERT model (`dslim/bert-base-NER`). I used pre-trained on NER dataset BERT model from the Hugging Face and fine-tuned it on the artificially created dataset. Model is able to identify B-ANIMAL (beginning of the entity), I-ANIMAL (inside of the entity) and O (outside of the entity) tokens.
The NER model was saved to Hugging Face model hub and could be accessed by the name `glekk/bert-base-ner-animals`.

The solution for image classification is based on the EfficientNetB3 model from PyTorch. I used the weights from the pre-trained model and fine-tuned it on the dataset containing 45 classes of animals. Considering the small size of the model, it was saved to the `models/image-model` folder and could be accessed from there.
Dataset for image classification was taken from Kaggle: [link](https://www.kaggle.com/datasets/asaniczka/mammals-image-classification-dataset-45-animals/).

You can find more information about the solution in the Jupiter notebooks in the `notebooks` folder.

## Project Structure 
- `data` folder - contains data for training and images for testing the pipeline
- `models` folder - contains the saved final models (only image classification model is included)
- `notebooks` folder - contains the notebooks with dataset creation and demonstration of the solution
- `src` folder - contains the source code
    - `train_image.py` - script for training the image model
    - `inference_image.py` - script for inference on the image model
    - `train_ner.py` - script for training the NER model
    - `inference_ner.py` - script for inference on the NER model
    - `utils.py` - utility functions
    - `pipeline.py` - script for the whole pipeline
- `.env` (included as .env.example) - environment variables file
- `requirements.txt` - requirements file

## Prerequisites
- Python 3.10 (it was tested on this version)
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
- Before all you need to set up the environment variables in .env file, there is .env.example as an example.
- You could change TEXT_MODEL_PATH to `glekk/bert-base-ner-animals` to directly access the models I've trained or train model by yourself.
- To run the training you need to run train_image.py/train_ner.py in src folder (you can run it from different folder, but then you may need to change paths in .env):
```
cd src
python train_image.py 
or
python train_ner.py
```
Also train scripts could be run with additional arguments that basically the same as in .env file (use -h to see them).
- Same for inference.py but they have some required arguments (check with -h):
```
cd src
python inference_image.py --image_path <path_to_image>
or
python inference_ner.py --text <text>
```
- To run the whole pipeline you need to run pipeline.py in src folder. It also has two required arguments --text and --image_path:
```
cd src
python pipeline.py --text <text> --image_path <path_to_image>
```
- To run the notebooks you just need to open them in appropriate environment (Jupiter Notebook\VS Code) and run all cells.
