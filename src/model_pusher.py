import mlflow
import mlflow.transformers
import dagshub
import logging
from transformers import pipeline
from utils.model_utils import get_best_f1
from config.constant import model_name, training_args

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

    def updated_model_pusher(self, trainer, metrics):
        try:
            new_f1 = metrics["eval_f1"]
            old_f1 = get_best_f1(self.experiment_name)
            print(f"New F1: {new_f1}"), 
            print(f"Old F1: {old_f1}")
            if old_f1 is None or new_f1 > old_f1:
                with mlflow.start_run():
                    #log the metrics
                    mlflow.log_metrics("accuracy", metrics["eval_accuracy"])
                    mlflow.log_metrics("f1", new_f1)

                    #log the parameters
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("epochs", training_args.num_train_epochs)
                    mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
                    mlflow.log_param("eval_batch_size", training_args.per_device_eval_batch_size)

                    #create a pipeline
                    sentiment_pipeline = pipeline(
                        task = "text-classification", 
                        model = trainer.model, 
                        tokenizer = model_name,
                        return_all_scores = True
                    )
                    
                    #log the model with the tokenizer
                    mlflow.transformers.log_model(
                        transformers_model = sentiment_pipeline,
                        artifact_path = "model",
                        registered_model_name = model_name
                    )
                logging.info("Model and metrics have been successfully pushed to MLflow successfully.")
            else:
                logging.info("Model not pushed to mlflow, New model did not outperform the existing model.")                    
 

        except Exception as e:
            logging.error(f"Error occurred while updating model pusher: {e}")
        