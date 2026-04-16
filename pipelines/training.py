import pandas as pd
from src.data_preprocessing import prepare_sentiment_data
from src.model_training import Training
from src.model_pusher import ModelPusher
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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
        
        print(results)
        pusher = ModelPusher()
        pusher.updated_model_pusher(trainer, results)

    except Exception as e:
        logging.error(f"Error occurred during training and evaluation: {e}") 

train_and_evaluate()