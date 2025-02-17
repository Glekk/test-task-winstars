import os
import logging
from argparse import ArgumentParser
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict
import evaluate
import numpy as np
from utils import set_seed

# Load constants from .env file
load_dotenv()
SEED = int(os.getenv('SEED'))
TEXT_DATASET_RAW_PATH = os.getenv('TEXT_DATASET_RAW_PATH')
MODEL_HUGGINGFACE_NAME = os.getenv('MODEL_HUGGINGFACE_NAME')
TEXT_MODEL_PATH = os.getenv('TEXT_MODEL_PATH')
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL')
LOGGING_FORMAT = os.getenv('LOGGING_FORMAT')
LOGGING_DATE_FORMAT = os.getenv('LOGGING_DATE_FORMAT')
LEARNING_RATE_TEXT_MODEL = float(os.getenv('LEARNING_RATE_TEXT_MODEL'))
EPOCHS_TEXT_MODEL = int(os.getenv('EPOCHS_TEXT_MODEL'))
BATCH_SIZE_TEXT_MODEL = int(os.getenv('BATCH_SIZE_TEXT_MODEL'))


def tokenize_and_align_labels(examples, tokenizer):
    '''
    Tokenize the examples and align the labels with the tokenized inputs.

    Args:
        examples (dict): The examples to tokenize.
        tokenizer (transformers.Tokenizer): The tokenizer to use.
    '''
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # -100 is the index to ignore.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                temp_label = label[word_idx]
                if temp_label % 2 == 1: # If B-ANIMAL is repeated, change it to I-ANIMAL.
                    temp_label += 1
                label_ids.append(temp_label)

            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def compute_metrics_extra_args(metric, id2label):
    '''
    Compute the metrics for the model.

    Outer Args:
        metric (evaluate.Metric): The metric to use.
        id2label (dict): The mapping of the label ids to the labels.

    Inner Args:
        eval_preds (tuple): The predictions from the model (logits, labels).

    Returns:
        dict: The metrics for the model (precision, recall, f1, accuracy).
    ''' 
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index and convert ids to labels
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            'precision': all_metrics['overall_precision'],
            'recall': all_metrics['overall_recall'],
            'f1': all_metrics['overall_f1'],
            'accuracy': all_metrics['overall_accuracy'],
        }
    return compute_metrics


def get_text_dataset(path: str):
    '''
    Get the text dataset from the path.

    Args:
        path (str): The path to the text dataset.

    Returns:
        dict: The text dataset.
    '''
    dataset_path_train = os.path.join(path, 'animal_ner_dataset_train.json')
    dataset_path_test = os.path.join(path, 'animal_ner_dataset_test.json')

    dataset = load_dataset('json', data_files={'train': dataset_path_train, 'test': dataset_path_test})
    return dataset

def main(seed: int = None, path: str = None, model_huggingface_name: str = None, 
         batch_size: int = None, learning_rate: float = None, epochs: int = None, model_save_path: str = None) -> None:
    '''
    Train a text model

    Args:
        seed (int, optional): seed for reproducibility
        path (str, optional): path to the dataset
        model_huggingface_name (str, optional): name of the Hugging Face model to use
        batch_size (int, optional): batch size for the DataLoader object
        learning_rate (float, optional): learning rate for the optimizer
        epochs (int, optional): number of epochs to train the model
        model_save_path (str, optional): path to save the model
    '''
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)

    # Check if the arguments are provided, otherwise use the environment variables
    seed = seed if seed else SEED
    path = path if path else TEXT_DATASET_RAW_PATH
    model_huggingface_name = model_huggingface_name if model_huggingface_name else MODEL_HUGGINGFACE_NAME
    batch_size = batch_size if batch_size else BATCH_SIZE_TEXT_MODEL
    learning_rate = learning_rate if learning_rate else LEARNING_RATE_TEXT_MODEL
    epochs = epochs if epochs else EPOCHS_TEXT_MODEL
    model_save_path = model_save_path if model_save_path else TEXT_MODEL_PATH

    #set seed for reproducibility
    set_seed(seed)

    dataset = get_text_dataset(path)
    logging.info('Dataset loaded')

    tokenizer = AutoTokenizer.from_pretrained(model_huggingface_name)
    logging.info('Tokenizer loaded')

    tokenized_ds = dataset.map(tokenize_and_align_labels, 
                      batched=True,
                      remove_columns=dataset['train'].column_names,
                      fn_kwargs={'tokenizer': tokenizer})
    logging.info('Dataset tokenized')

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    logging.info('Data collator created')

    # Creating maps for the labels and ids
    label_names = ['O', 'B-ANIMAL', 'I-ANIMAL']
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {label: i for i, label in label2id.items()}

    model = AutoModelForTokenClassification.from_pretrained(model_huggingface_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    logging.info('Model loaded')

    training_args = TrainingArguments(
        output_dir=model_save_path,
        eval_strategy='epoch',
        save_strategy='no',
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_strategy='epoch',
        seed=SEED,
    )

    metric = evaluate.load('seqeval')

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['test'],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_extra_args(metric, id2label)
    )
    logging.info('Trainer created')

    trainer.train()
    logging.info('Model trained')

    trainer.save_model(model_save_path)
    logging.info('Model saved')

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, help='Seed for reproducibility')
    parser.add_argument('--path', type=str, help='Path to the dataset')
    parser.add_argument('--model_huggingface_name', type=str, help='Name of the Hugging Face model to use')
    parser.add_argument('--batch_size', type=int, help='Batch size for the DataLoader object')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train the model')
    parser.add_argument('--model_save_path', type=str, help='Path to save the model')

    args = parser.parse_args()

    seed = args.seed
    path = args.path
    model_huggingface_name = args.model_huggingface_name
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    model_save_path = args.model_save_path

    main(seed, path, model_huggingface_name, batch_size, learning_rate, epochs, model_save_path)
