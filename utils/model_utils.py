import dagshub
import mlflow
import mlflow.tracking import MlflowClient
import logging

dagshub.init(repo_owner='isiakpereaghogho', repo_name='Sentiment_Analysis_for_ShopEase_Customer_Feedback', mlflow=True)

logging.basicConfig(level=logging.INFO)

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