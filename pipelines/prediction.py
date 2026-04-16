from utils.model_utils import load_registered_model
from src.data_cleaning import DataCleaning
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class predict_sentiment:
    def __init__(self):
        self.pipelines = load_registered_model()
        #define the label mapping
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, text):
        raw_result = self.pipelines(text)
        # Map label to actual text label
        for item in raw_result:
            index = int(item['label'].split('_')[-1])  # Extract index from label like 'LABEL_0'
            item['label'] = self.id2label.get(index, item['label'])  # Map
            
        return raw_result