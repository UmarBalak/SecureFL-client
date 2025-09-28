import os
import glob
import time
import logging
import tempfile
import json
from datetime import datetime
from typing import List, Optional, Tuple, Any, Dict
import re
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.client')
BLOB_SERVICE_CLIENT = None

if 'CLIENT_ACCOUNT_URL' in os.environ:
    CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
    CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")
    if not CLIENT_ACCOUNT_URL:
        raise ValueError("Missing required environment variable: Account url")
    try:
        BLOB_SERVICE_CLIENT = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
    except Exception as e:
        print(f"Failed to initialize Azure Blob Service: {e}")
        raise


def upload_file(file_path, container_name, metadata):
    filename = os.path.basename(file_path)
    print(f"Uploading weights ({filename}) to Azure Blob Storage...")
    try:
        blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container_name, blob=filename)
        with open(file_path, "rb") as file:
            blob_client.upload_blob(file.read(), overwrite=True, metadata=metadata)
        print(f"Weights ({filename}) uploaded successfully to Azure Blob Storage.")
    except Exception as e:
        print(f"Error uploading weights ({filename}): {e}")

def upload_json_to_blob(data: Dict[str, Any], filename: str, container_name: str, metadata: Dict[str, str]) -> bool:
    """
    Serialize a Python dictionary to JSON and upload it to Azure Blob Storage.

    Args:
        data:        Dictionary to serialize as JSON.
        filename:    Name of the blob (e.g., 'metrics.json').
        metadata:    Key–value metadata to attach to the blob.

    Returns:
        bool: True on success, False on failure.
    """
    temp_path = None
    try:
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_path = temp_file.name
            json.dump(data, temp_file, ensure_ascii=False, indent=2)

        # Upload JSON file to Azure Blob Storage
        blob_client =BLOB_SERVICE_CLIENT.get_blob_client(
            container=container_name,
            blob=filename
        )
        with open(temp_path, "rb") as file:
            blob_client.upload_blob(file, overwrite=True, metadata=metadata)

        print(f"Successfully uploaded JSON to blob: {filename}")
        return True

    except Exception as e:
        print(f"Error uploading JSON to blob: {e}")
        return False

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def save_run_info(config, model_info, eval_results):
    run_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': config,
        'model_info': model_info,
        'evaluation_results': {k: v for k, v in eval_results.items()
                              if k != 'confusion_matrix' and k != 'per_class_metrics'}
    }
    os.makedirs('logs', exist_ok=True)
    with open('logs/run_summary.json', 'w') as f:
        json.dump(run_info, f, indent=2)
    print("Run information saved to 'logs/run_summary.json'")

def find_csv_file(file_pattern):
    matching_files = glob.glob(file_pattern)
    if matching_files:
        print(f"Found dataset: {matching_files[0]}")
        return matching_files[0]
    else:
        print(f"No dataset found matching pattern: {file_pattern}")
        return None

def wait_for_csv(file_pattern, wait_time=300):
    print(f"Checking for dataset matching pattern: {file_pattern}")
    while True:
        csv_file = find_csv_file(file_pattern)
        if csv_file:
            return csv_file
        print(f"Dataset not found. Waiting for {wait_time // 60} minutes...")
        time.sleep(wait_time)
        print(f"Rechecking for dataset matching pattern: {file_pattern}")

def load_model_weights(model, directory_path):
    try:
        # List all .h5 files matching pattern g<number>.h5
        weight_files = glob.glob(os.path.join(directory_path, "g*.h5"))
        if not weight_files:
            print(f"No matching .h5 files found in {directory_path}")
            return False, model
        
        # Extract numeric suffix from filenames and find max
        def extract_num(filename):
            match = re.search(r'g(\d+)\.h5$', filename)
            return int(match.group(1)) if match else -1
        
        weight_files.sort(key=extract_num)
        highest_weight_file = weight_files[-1]

        print(f"Loading weights from {highest_weight_file}")
        model.load_weights(highest_weight_file)
        print(f"Successfully loaded weights from {highest_weight_file}")
        return True, model
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        return False, model

def get_versioned_filename(client_id, save_dir, extension=".h5"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_pattern = re.compile(rf"client{client_id}_v(\d+).*\.{extension}")
    existing_versions = [
        int(version_pattern.match(f).group(1))
        for f in os.listdir(save_dir)
        if version_pattern.match(f)
    ]
    next_version = max(existing_versions, default=0) + 1
    filename = f"client{client_id}_v{next_version}_{timestamp}.{extension}"
    return os.path.join(save_dir, filename), next_version, timestamp

def get_versioned_metadata_filename(client_id, save_dir, extension=".json"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_pattern = re.compile(rf"client{client_id}_v(\d+).*\.{extension}")
    existing_versions = [
        int(version_pattern.match(f).group(1))
        for f in os.listdir(save_dir)
        if version_pattern.match(f)
    ]
    next_version = max(existing_versions, default=0) + 1
    filename = f"client{client_id}_v{next_version}_{timestamp}.{extension}"
    return filename

def save_weights(client_id, model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    weights_path, next_version, timestamp = get_versioned_filename(client_id, save_dir, extension="h5")
    try:
        model.save_weights(weights_path)
        print(f"Weights for {client_id} saved at {weights_path}")
    except Exception as e:
        print(f"Failed to save weights for {client_id}: {e}")
    return weights_path, timestamp

def save_model(client_id, model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"weights.h5")
    try:
        model.save(model_path)
        print(f"Model for {client_id} saved at {model_path}")
    except Exception as e:
        print(f"Failed to save model for {client_id}: {e}")
    return model_path