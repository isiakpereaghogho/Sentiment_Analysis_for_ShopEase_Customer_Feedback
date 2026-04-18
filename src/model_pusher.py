import mlflow
import mlflow.transformers
import dagshub
import logging
from transformers import pipeline
from utils.model_utils import get_best_f1
from config.constant import model_name, training_args, registered_model_name

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelPusher:
    def __init__(self, experiment_name="sentiment_analysis_experiment"):
        try:
            dagshub.init(
                repo_owner='isiakpereaghogho',
                repo_name='Sentiment_Analysis_for_ShopEase_Customer_Feedback',
                mlflow=True
            )
            self.experiment_name = experiment_name
            mlflow.set_experiment(self.experiment_name)
            logging.info("ModelPusher has been successfully initialized.")
        except Exception as e:
            logging.error(f"Error occurred while initializing ModelPusher: {e}")
            raise

    def updated_model_pusher(self, trainer, metrics):
        try:
            new_f1 = metrics["eval_f1"]
            old_f1 = get_best_f1(self.experiment_name)

            print(f"New F1: {new_f1}")
            print(f"Old F1: {old_f1}")

            if old_f1 is None or new_f1 >= old_f1:
                with mlflow.start_run(run_name="sentiment_model_training"):
                    mlflow.log_metric("accuracy", metrics["eval_accuracy"])
                    mlflow.log_metric("f1", new_f1)

                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("epochs", training_args.num_train_epochs)
                    mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
                    mlflow.log_param("eval_batch_size", training_args.per_device_eval_batch_size)

                    sentiment_pipeline = pipeline(
                            task="text-classification",
                            model=trainer.model,
                            tokenizer=model_name,
                            top_k=None
                    )

                    mlflow.transformers.log_model(
                    transformers_model=sentiment_pipeline,
                    name="model",
                    registered_model_name=registered_model_name,
                    # signature=False,
                    # pip_requirements=[
                    #         "mlflow",
                    #         "transformers",
                    #         "torch",
                    #         "pandas",
                    #         "numpy",
                    #         "accelerate"
                    #         ],
                    await_registration_for=0
                    )

                logging.info("Model and metrics have been successfully pushed to MLflow.")
                return True
            else:
                logging.info("Model not pushed to MLflow, new model did not outperform the existing model.")
                return False

        except Exception as e:
            logging.error(f"Error occurred while updating model pusher: {e}")
            raise