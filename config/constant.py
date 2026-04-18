import os

from transformers import Trainer, TrainingArguments

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Input_Data = os.path.join(BASE_DIR, "Data", "sample_reviews.csv")
Cleaned_Data = os.path.join(BASE_DIR, "Data", "cleaned_data", "clean_reviews.csv")
Train_Data = os.path.join(BASE_DIR, "Data", "processed_data", "train_data.pt")
Test_Data = os.path.join(BASE_DIR, "Data", "processed_data", "test_data.pt")

model_name = "distilbert-base-multilingual-cased"
registered_model_name = "distilbert_sentiment_model"
num_labels =3
max_length=128
padding=True
truncation=True
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)