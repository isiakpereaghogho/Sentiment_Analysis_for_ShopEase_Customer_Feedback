import dagshub
import mlflow
from mlflow.tracking import MlflowClient
import logging
from config.constant import model_name

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

def get_best_model(experiment_name = "sentiment_analysis_experiment"):
    try:
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
        
def get_best_f1(experiment_name = "sentiment_analysis_experiment"):
    best_model = get_best_model(experiment_name)
    if best_model is None:
        return None
    return best_model.data.metrics.get("f1", 0)

def load_registered_model(model_name = model_name):
    
    model_uri = f"models:/{model_name}/latest"
    sentiment_pipeline = mlflow.transformers.load_model(model_uri)
    return sentiment_pipeline