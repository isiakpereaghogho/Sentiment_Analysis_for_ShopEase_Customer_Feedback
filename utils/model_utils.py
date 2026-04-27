import dagshub
import mlflow
from mlflow.tracking import MlflowClient
import logging
from config.constant import registered_model_name
import os

logging.basicConfig(level=logging.INFO)

def init_dagshub():
    try:
        dagshub.init(
            repo_owner='isiakpereaghogho',
            repo_name='Sentiment_Analysis_for_ShopEase_Customer_Feedback',
            mlflow=True
        )
        logging.info("DagsHub initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize DagsHub: {e}")
        raise

def get_best_model(experiment_name="sentiment_analysis_experiment"):
    try:
        init_dagshub()
        client = MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        runs = client.search_runs([experiment.experiment_id])
        if not runs:
            return None

        best_model = sorted(
            runs,
            key=lambda x: x.data.metrics.get("f1", 0),
            reverse=True
        )[0]
        return best_model
    except Exception as e:
        logging.error(f"Error occurred while fetching the best model: {e}")
        return None

def get_best_f1(experiment_name="sentiment_analysis_experiment"):
    best_model = get_best_model(experiment_name)
    if best_model is None:
        return None
    return best_model.data.metrics.get("f1", 0)

def load_registered_model(registered_name=registered_model_name):
    # Local testing initialization of DagsHub and MLflow tracking URI
    #dagshub.init(
    #    repo_owner='isiakpereaghogho',
    #    repo_name='Sentiment_Analysis_for_ShopEase_Customer_Feedback',
    #    mlflow=True
    #)

    #Remote / Production access to mlflow dagshub
    dagshub_token = os.getenv("ShopEase_env_Dagshub_token")
    if not dagshub_token:
        raise ValueError("Dagshub token not found in environment variables.")
            
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "isiakpereaghogho"
    repo_name = "Sentiment_Analysis_for_ShopEase_Customer_Feedback"

    #Setting up the MLflow tracking URI to point to Dagshub
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    model_uri = f"models:/{registered_name}/11"

    sentiment_pipeline = mlflow.transformers.load_model(model_uri)

    return sentiment_pipeline
import time
import mlflow.transformers

def load_registered_model_with_retry(model_uri, retries=5, delay=20):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(f"Loading model attempt {attempt}/{retries}")
            return mlflow.transformers.load_model(model_uri)
        except Exception as e:
            last_error = e
            print(f"Model load failed: {e}")
            time.sleep(delay)

    raise last_error
    