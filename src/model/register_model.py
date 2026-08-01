# register model

import json
import mlflow
import logging
import os
import dagshub
from dotenv import load_dotenv
import pickle

# Load environment variables from .env file
load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token



dagshub_url = "https://dagshub.com"
repo_owner = "iamdebasishdas123"
repo_name = "YouTube-Mood-Tracker"



# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    client = mlflow.tracking.MlflowClient()
    model_path = model_info['model_path']+".pkl"
    # Load the model from the local pickle file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.debug('Model loaded from %s', model_info['model_path'])

    # Set the correct experiment (avoid defaulting to experiment 0)
    

    # Log AND register inside the same run context
    with mlflow.start_run(run_name="model_registration") as run:
        model_info_mlflow = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",        
            registered_model_name=model_name  # ✅ register directly here
        )
        new_run_id = run.info.run_id
        logger.debug('Model logged and registered in run %s', new_run_id)

    # Transition to Staging
    # Get the latest version just registered
    versions = client.get_latest_versions(model_name, stages=["None"])
    latest_version = versions[0].version

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Staging"
    )
    logger.debug(
        'Model %s version %s transitioned to Staging.',
        model_name, latest_version
    )


def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "yt_chrome_plugin_model"  # You can also get this from model_info if it's stored there
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()