from transformers import AutoTokenizer, BertTokenizer
import pandas as pd
import numpy as np
import logging
import torch
import os
from sklearn.model_selection import train_test_split
from config.constant import Input_Data, model_name, truncation, padding, max_length, Train_Data, Test_Data
from src.data_cleaning import DataCleaning
from src.data_ingestion import data_ingestion






class data_processor:
    def __init__(self):
        self.sentiment_data = data_ingestion()
        self.cleaner = DataCleaning()
        self.data = self.cleaner.clean_data(self.sentiment_data)

    def split_data(self):
        try:
            X = self.data['final_review'].astype(str)
            y = self.data['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            logging.info("Data splitted successfully.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error occurred while splitting the data: {e}")

class Tokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode(self, text):
        return self.tokenizer(
            text.to_list(),padding=padding,truncation=truncation, max_length=max_length)
    
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        #Ensuring labels are list to avoid pandas index issues
        if hasattr(labels, 'tolist'):
            self.labels = labels.tolist() # Convert pandas series to list
        elif hasattr(labels, '__iter__') and not isinstance(labels, (list, tuple)):
            self.labels = list(labels)
        else:
            self.labels = labels

    def __len__(self):
        return len(self.labels)
              
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
            
def prepare_sentiment_data():
    try:
        #call data processor class to split data
        processor = data_processor()
        #split the data
        X_train, X_test, y_train, y_test = processor.split_data()
        # Converting labels to list to avoid pandas index issues
        if hasattr(y_train, 'tolist'):
            y_train = y_train.tolist()
        if hasattr(y_test, 'tolist'):
            y_test = y_test.tolist()
        # call tokenizer to encode the X_train and X_test        
        tokenizer = Tokenizer()
        train_encodings = tokenizer.encode(X_train)
        test_encodings = tokenizer.encode(X_test)
        # Convert the train encodings and test encodings to pytorch dataset using the SentimentDataset class
        train_dataset = SentimentDataset(train_encodings, y_train)
        test_dataset = SentimentDataset(test_encodings, y_test)
        logging.info("Dataset has been prepared successfully.")
        # Save the train dataset and test dataset as .pt files into the  directory specified in the constant file
        os.makedirs(os.path.dirname(Train_Data), exist_ok=True)
        os.makedirs(os.path.dirname(Test_Data), exist_ok=True)
        torch.save(train_dataset, Train_Data)
        torch.save(test_dataset, Test_Data) 
        return train_dataset, test_dataset
        
    except Exception as e: 
        logging.error(f"error occurred while preparing sentiment data: {e}")


