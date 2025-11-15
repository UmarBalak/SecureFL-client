import os
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import clean preprocessor (no download logic)
from preprocessing_client import IoTDataPreprocessorClient
# from training_client import IoTModelTrainer
from training_custom_gaussian import IoTModelTrainer
from evaluate import evaluate_model
from functions import wait_for_csv, load_model_weights, upload_file, save_weights, upload_json_to_blob, get_versioned_metadata_filename
from dotenv import load_dotenv

# Configuration
DATASET_PATH = "./DATA/global_train.csv"
TEST_DATASET_PATH = "./DATA/global_test.csv"
ARTIFACTS_PATH = "artifacts"  # Websocket service downloads here

script_directory = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(script_directory, "models")
GLOBAL_MODEL_PATH = os.path.join(script_directory, "GLOBAL_MODELS")

CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")

load_dotenv(dotenv_path=".env.client")

def main(client_id, epochs=20):
    """
    Clean client main - assumes artifacts already downloaded by websocket service
    """
    config = {
        'data_path': wait_for_csv(DATASET_PATH),
        'test_data_path': wait_for_csv(TEST_DATASET_PATH),
        'epochs': epochs,
        'batch_size': 256,
        'random_state': 42,
        'model_architecture': [256, 256],  # Match global
        'learning_rate': 0.001,  # global = 5e-5 / better for client (tested) = 0.0001
        'num_classes': 15,
        "l2_norm_clip": 3.0,
        "noise_multiplier": 1.0,
        'delta': 1e-5,
    }
    
    # Set random seeds
    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])
    
    # Create directories
    for directory in ['models', 'logs', 'plots', 'data', 'federated_models']:
        os.makedirs(directory, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"SecureFL Client {client_id} - Clean Integration")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Artifacts path: {ARTIFACTS_PATH}")
    print(f"{'='*70}\n")
    
    # --------------------------
    # 1) Initialize preprocessor (artifacts already downloaded)
    # --------------------------
    print("🔧 Initializing preprocessor with server artifacts...")
    try:
        preprocessor = IoTDataPreprocessorClient(artifacts_path=ARTIFACTS_PATH)
    except Exception as e:
        print(f"❌ Preprocessing initialization failed: {e}")
        print("💡 Make sure websocket_service.py has downloaded server artifacts!")
        print("💡 Required files: preprocessor.pkl, global_label_encoder.pkl, feature_info.pkl")
        raise
    
    # --------------------------
    # 2) Preprocess client data
    # --------------------------
    print("🔧 Preprocessing client data...")
    X_train, y_train, num_classes_train = preprocessor.preprocess_data(config['data_path'])
    X_test, y_test, num_classes_test = preprocessor.preprocess_data(config['test_data_path'])
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=config['random_state'], 
        stratify=y_train if len(np.unique(y_train)) > 1 else None
    )
    
    print(f"✅ Data ready: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    if len(X_train) > 5000:
        config['batch_size'] = 512
    
    # --------------------------
    # 3) Build model with server dimensions
    # --------------------------
    trainer = IoTModelTrainer(random_state=config['random_state'])
    model = trainer.create_model(
        input_dim=X_train.shape[1],  # Should match server dimensions
        num_classes=config['num_classes'],
        architecture=config['model_architecture']
    )
    
    print(f"✅ Model architecture: {X_train.shape[1]} → {config['model_architecture']} → {config['num_classes']}")
    
    # Load global model weights (downloaded by websocket service)
    # is_success, model = load_model_weights(model, GLOBAL_MODEL_PATH)
    # print("✅ Global model loaded" if is_success else "⚠️ Training from scratch")
    
    # --------------------------
    # 4) Train model
    # --------------------------
    y_train_cat = to_categorical(y_train, num_classes=config['num_classes'])
    y_val_cat = to_categorical(y_val, num_classes=config['num_classes'])
    y_test_cat = to_categorical(y_test, num_classes=config['num_classes'])
    
    # Train with differential privacy
    # history, training_time, num_samples, delta, final_epsilon = trainer.train_model(
    #     X_train, y_train_cat, X_val, y_val_cat,
    #     model=model,
    #     architecture=config['model_architecture'],
    #     epochs=config['epochs'],
    #     batch_size=config['batch_size'],
    #     verbose=2,
    #     use_dp=True,
    #     use_custom_dp=True,
    #     l2_norm_clip=config['l2_norm_clip'],
    #     noise_multiplier=config['noise_multiplier'],
    #     learning_rate=config['learning_rate']
    # )
    # model = trainer.get_model()
    history, dp_perf, final_eps = trainer.train_model(
                X_train, y_train_cat, X_val, y_val_cat,
                model=model,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                verbose=2,
                use_dp=True,
                noise_type="gaussian",
                l2_norm_clip=config['l2_norm_clip'],
                noise_multiplier=config['noise_multiplier'],
                delta=config['delta'],
                learning_rate=config['learning_rate']
            )
    
    print("✅ Client training complete!")
    print(f"Final epsilon = {final_eps}")
    
    # --------------------------
    # 5) Evaluate
    # --------------------------
    try:
        class_names = preprocessor.global_le.classes_.tolist()  # ✅ Correct attribute
        print(f"✅ Class names loaded: {class_names}")
    except AttributeError:
        # Fallback to hardcoded class names
        class_names = [
            'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
            'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
            'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
        ]
        print(f"⚠️ Using fallback class names: {len(class_names)} classes")

    # model.compile(
    #         optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
    #         loss='categorical_crossentropy',
    #         metrics=[
    #             'accuracy'
    #         ]
    #     )
    
    eval_results = evaluate_model(model, X_test, y_test_cat, class_names=class_names)
    test_metrics = eval_results['test']
    
    print(f"✅ Evaluation complete:")
    print(f"   Test Loss: {test_metrics['loss']:.4f}")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test Macro F1: {test_metrics['macro_f1']:.4f}")
    
    # --------------------------
    # 6) Save and upload results
    # --------------------------
    # weights_path, timestamp = save_weights(client_id, model, SAVE_DIR)
    
    # Feature info for metadata
    # feature_info = preprocessor.get_feature_info()
    
    # metadata = {
    #             "final_test_loss": str(test_metrics['loss']),
    #             "final_test_accuracy": str(test_metrics['accuracy']),
    #             "final_test_precision": str(test_metrics['macro_precision']),
    #             "final_test_recall": str(test_metrics['macro_recall']),
    #             "final_test_f1": str(test_metrics['macro_f1']),
    #             "num_training_samples": str(num_samples),
    #         }

    # complete_metadata = {
    #     "test_metrics": test_metrics,
    #     "num_training_samples": str(num_samples),
    #     "data_classes_present": int(num_classes_train),
    #     "batch_size": config['batch_size'],
    #     "learning_rate": config['learning_rate'],
    #     "differential_privacy": True,
    #     "noise_multiplier": config['noise_multiplier'],
    #     "final_epsilon": final_epsilon,
    #     "delta": delta
    # }

    # upload_file(weights_path, CLIENT_CONTAINER_NAME, metadata)

    # metadata_filename = get_versioned_metadata_filename(client_id, SAVE_DIR)
    # uploaded_metadata = upload_json_to_blob(complete_metadata, metadata_filename, CLIENT_CONTAINER_NAME, {})
    
    print(f"\n✅ Client {client_id} completed with server preprocessing!")
    return eval_results

if __name__ == "__main__":
    CLIENT_ID = os.getenv("CLIENT_ID")
    main(CLIENT_ID, epochs=20)
