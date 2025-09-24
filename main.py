import os
import time
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
import threading
from statistics import mean
from itertools import product
import threading
from statistics import mean
import gc
import glob
import re
import logging
import pickle
import sys
import time, json, csv, psutil
from itertools import product
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import IoTDataPreprocessor
from model import IoTModel
from training import IoTModelTrainer
from evaluate import evaluate_model
from azure.storage.blob import BlobServiceClient # type: ignore

from functions import wait_for_csv, load_model_weights, upload_file, save_run_info, find_csv_file, save_model, save_weights


DATASET_PATH = "./DATA/client1.csv"
TEST_DATASET_PATH = "./DATA/test.csv"
DOTENV_PATH = ".env.client"
script_directory = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(script_directory, "models")
GLOBAL_MODEL_PATH = os.path.join(script_directory, "GLOBAL_MODELS")


from dotenv import load_dotenv
load_dotenv(dotenv_path=DOTENV_PATH)

CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")

if not CLIENT_ACCOUNT_URL:
    raise ValueError("Missing required environment variable: Account url")

try:
    BLOB_SERVICE_CLIENT = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
except Exception as e:
    print(f"Failed to initialize Azure Blob Service: {e}")
    raise


def main(client_id, epochs=20):
    config = {
        'data_path_pattern': DATASET_PATH,
        'test_data_path_pattern': TEST_DATASET_PATH,
        'epochs': epochs,
        'batch_size': 256,
        'random_state': 42,
        'model_architecture': [1024, 512, 256],
    }
    config['data_path'] = wait_for_csv(config['data_path_pattern'])
    config['test_data_path'] = wait_for_csv(config['test_data_path_pattern'])
    
    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])
    
    for directory in ['models', 'logs', 'plots', 'data', 'federated_models']:
        os.makedirs(directory, exist_ok=True)
        
    print(f"\n{'='*70}")
    print(f"SecureFL")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    preprocessor = IoTDataPreprocessor()

    ## after feature selection - Process training data
    X_train, y_train, num_classes = preprocessor.preprocess_data(
        config['data_path']
        )
    

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=config['random_state'], stratify=y_train
        )
    
    # Process test data separately
    print("Loading and preprocessing test data...")
    X_test, y_test, _ = preprocessor.preprocess_data(
        config['test_data_path']
        )

    trainer = IoTModelTrainer(random_state=config['random_state'])
    model = trainer.create_model(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        architecture=config['model_architecture']
    )
    print("@" * 50)
    print(f"Training data features: {X_train.shape[1]}")
    print(f"Test data features: {X_test.shape[1]}")
    print("@" * 50)

    is_success, model = load_model_weights(model, GLOBAL_MODEL_PATH)
    if is_success:
        print("Weights loaded successfully.")
    else:
        print("Failed to load weights. Training from scratch.")


    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    use_dp = True
    l2_norm_clip = 3.0
    noise_multiplier = 1.2
    microbatches = 1

    history, training_time, num_samples, delta, epsilon_dict = trainer.train_model(
        X_train, y_train_cat, X_val, y_val_cat,
        model=model,
        architecture=config['model_architecture'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=2,
        use_dp=use_dp,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        microbatches=microbatches,
        learning_rate=0.0001
    )
    
    model = trainer.get_model()
    print("Model training complete.")
    model.summary()

    # Get class names from the preprocessor's LabelEncoder
    le = preprocessor.le_dict.get('Attack_type', None)
    class_names = le.classes_.tolist() if le else None
    
    # Evaluate model on all sets
    eval_results = evaluate_model(model, X_test, y_test_cat, class_names=class_names)
    test_metrics = eval_results['test']

    # Save model and weights
    weights_path, timestamp = save_weights(client_id, model, SAVE_DIR)

    # Prepare metadata for Azure upload
    metadata = {        
        "final_test_loss": str(test_metrics['loss']),
        "final_test_accuracy": str(test_metrics['accuracy']),
        "final_test_precision": str(test_metrics['macro_precision']),
        "final_test_recall": str(test_metrics['macro_recall']),
        "final_test_f1": str(test_metrics['macro_f1']),

        "num_training_samples": str(num_samples),
        "epochs": str(epochs),
        "batch_size": str(config['batch_size']),
        "model_architecture": json.dumps(config['model_architecture']),

        "l2_norm_clip": str(l2_norm_clip),
        "noise_multiplier": str(noise_multiplier),
        "microbatches": str(microbatches),

        "delta": str(delta),
        "epsilon_dict": json.dumps(epsilon_dict) # Epsilon values for each checkpoint epoch.
    }
    
    # Upload model and weights to Azure
    upload_file(weights_path, CLIENT_CONTAINER_NAME, metadata)


import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv(dotenv_path=DOTENV_PATH)
    client_id = os.getenv("CLIENT_ID")
    if not client_id:
        raise ValueError("Missing required environment variable: CLIENT_ID")
    print(f"Client ID: {client_id}")

    main(client_id, epochs=20)