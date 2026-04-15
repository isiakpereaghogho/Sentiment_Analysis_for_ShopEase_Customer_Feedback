import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from config.constant import Train_Data, Test_Data, model_name, training_args, num_labels
from src.data_cleaning import DataCleaning
import logging
import numpy as np
from src.data_preprocessing import prepare_sentiment_data


class Training:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
         

    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {"accuracy": acc, "f1": f1}

    def model_training(self, train_dataset, test_dataset):
        try:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=self.compute_metrics
            )
            trainer.train()
            logging.info("Model training completed successfully.")
            return trainer
        except Exception as e:
            logging.error(f"Error occurred during model training: {e}") 

    def model_evaluation(self, trainer):
        results = trainer.evaluate()
        return results
    
def train_and_evaluate():
    try:
        # Prepare the dataset
        train_dataset, test_dataset =prepare_sentiment_data()

        # Initialize the training class
        train = Training()

        # Train the model
        trainer = train.model_training(train_dataset, test_dataset)

        # Evaluate the model
        results = train.model_evaluation(trainer)
        logging.info(f"Evaluation results: {results}")
    except Exception as e:
        logging.error(f"Error occurred during training and evaluation: {e}") 

train_and_evaluate()   