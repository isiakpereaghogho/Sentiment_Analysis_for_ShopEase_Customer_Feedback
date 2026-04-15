import mlflow
import mlflow.transformers
import dagshub
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelPusher:
    def __init__(self, experiment_name="sentiment_analysis_experiment"):
        try:
            dagshub.init(repo_owner = 'isiakpereaghogho', repo_name = 'Sentiment_Analysis_for_ShopEase_Customer_Feedback', mlflow=True)
            self.experiment_name = experiment_name 
            mlflow.set_experiment(self.experiment_name)     
            logging.info("ModelPusher has been successfully initialized.")
        except Exception as e:
            logging.error(f"Error occurred while initializing ModelPusher: {e}")