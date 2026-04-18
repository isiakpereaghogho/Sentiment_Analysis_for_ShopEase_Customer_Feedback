from utils.model_utils import load_registered_model
from src.data_cleaning import DataCleaning
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class PredictSentiment:
    def __init__(self):
        self.pipelines = load_registered_model()
        #define the label mapping
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, text):
        raw_result = self.pipelines(text)

        # Map labels and build scores dict
        scores = {}
        for item in raw_result:
            index = int(item['label'].split('_')[-1])
            label = self.id2label.get(index, item['label'])
            score = item['score']
            scores[label] = score

        # Get best label (highest score)
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        return {"sentiment": best_label,"confidence": best_score,"scores": scores}