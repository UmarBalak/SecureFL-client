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
from training_custom_gaussian import IoTModelTrainer
from evaluate import evaluate_model
from functions import wait_for_csv, load_model_weights, upload_file, save_weights, upload_json_to_blob, get_versioned_metadata_filename
from dotenv import load_dotenv
from compression_tflite import TFLiteCompressor
from compression_quantization import QuantizationCompressor

# Configuration
DATASET_PATH = "./DATA/global_train.csv"
TEST_DATASET_PATH = "./DATA/global_test.csv"
ARTIFACTS_PATH = "artifacts"  # Websocket service downloads here
script_directory = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(script_directory, "models")
GLOBAL_MODEL_PATH = os.path.join(script_directory, "GLOBAL_MODELS")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")

load_dotenv(dotenv_path=".env.client")

def compress_and_save_weights(client_id, model, X_train, save_dir,
                               compression_method='float16'):
    """
    Compress model using specified method and save
    
    Parameters:
    -----------
    client_id : str
        Client identifier
    model : tf.keras.Model
        Trained model
    X_train : np.array
        Training data for quantization
    save_dir : str
        Directory to save compressed models
    compression_method : str
        'float16' (recommended), 'tflite_dynamic', 'int8', or 'none'
    
    Returns:
    --------
    results : dict
        Compression results with file paths and ratios
    """
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get original model size for comparison
    temp_h5 = os.path.join(save_dir, f"temp_{client_id}.h5")
    model.save_weights(temp_h5)
    original_size_mb = os.path.getsize(temp_h5) / (1024 * 1024)
    os.remove(temp_h5)
    
    if compression_method == 'none':
        # Save as original H5 format (no compression)
        h5_path = os.path.join(save_dir, f"{client_id}_v1_{timestamp}.h5")
        model.save_weights(h5_path)
        
        results['primary'] = {
            'method': 'none',
            'path': h5_path,
            'compression_ratio': 1.0,
            'size_mb': original_size_mb,
            'original_size_mb': original_size_mb
        }
        print(f"✅ Saved original H5 weights: {original_size_mb:.2f} MB")
        
    elif compression_method == 'float16':
        # Float16 post-training quantization (RECOMMENDED)
        quant_compressor = QuantizationCompressor()
        quant_model_float16, ratio_float16 = quant_compressor.compress_model(
            model, quantization_type='float16'
        )
        
        quant_path = os.path.join(save_dir, f"{client_id}_v1_{timestamp}.tflite")
        quant_compressor.save_quantized_model(quant_model_float16, quant_path)
        
        compressed_size_mb = len(quant_model_float16) / (1024 * 1024)
        
        results['primary'] = {
            'method': 'float16',
            'path': quant_path,
            'compression_ratio': ratio_float16,
            'size_mb': compressed_size_mb,
            'original_size_mb': original_size_mb
        }
        print(f"✅ Float16 quantization: {ratio_float16:.2f}x reduction")
        print(f"   Original: {original_size_mb:.2f} MB → Compressed: {compressed_size_mb:.2f} MB")
        
    elif compression_method == 'tflite_dynamic':
        # TFLite with dynamic range quantization
        tflite_compressor = TFLiteCompressor()
        tflite_model, ratio = tflite_compressor.compress_model(
            model, optimization_mode='dynamic_range'
        )
        
        tflite_path = os.path.join(save_dir, f"{client_id}_v1_{timestamp}.tflite")
        tflite_compressor.save_compressed_model(tflite_model, tflite_path)
        
        compressed_size_mb = len(tflite_model) / (1024 * 1024)
        
        results['primary'] = {
            'method': 'tflite_dynamic',
            'path': tflite_path,
            'compression_ratio': ratio,
            'size_mb': compressed_size_mb,
            'original_size_mb': original_size_mb
        }
        print(f"✅ TFLite dynamic range: {ratio:.2f}x reduction")
        print(f"   Original: {original_size_mb:.2f} MB → Compressed: {compressed_size_mb:.2f} MB")
        
    elif compression_method == 'int8':
        # Int8 full quantization (requires calibration dataset)
        quant_compressor = QuantizationCompressor()
        rep_dataset = quant_compressor.create_representative_dataset(X_train, num_samples=100)
        quant_model_int8, ratio_int8 = quant_compressor.compress_model(
            model, quantization_type='int8', 
            representative_dataset=rep_dataset
        )
        
        quant_path = os.path.join(save_dir, f"{client_id}_v1_{timestamp}.tflite")
        quant_compressor.save_quantized_model(quant_model_int8, quant_path)
        
        compressed_size_mb = len(quant_model_int8) / (1024 * 1024)
        
        results['primary'] = {
            'method': 'int8',
            'path': quant_path,
            'compression_ratio': ratio_int8,
            'size_mb': compressed_size_mb,
            'original_size_mb': original_size_mb
        }
        print(f"✅ Int8 quantization: {ratio_int8:.2f}x reduction")
        print(f"   Original: {original_size_mb:.2f} MB → Compressed: {compressed_size_mb:.2f} MB")
        
    else:
        raise ValueError(f"Unknown compression method: {compression_method}")
    
    return results


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
    is_success, model = load_model_weights(model, GLOBAL_MODEL_PATH)
    print("✅ Global model loaded" if is_success else "⚠️ Training from scratch")

    # --------------------------
    # 4) Train model
    # --------------------------
    y_train_cat = to_categorical(y_train, num_classes=config['num_classes'])
    y_val_cat = to_categorical(y_val, num_classes=config['num_classes'])
    y_test_cat = to_categorical(y_test, num_classes=config['num_classes'])
    num_samples = len(X_train)

    # Train with differential privacy (custom)
    history, dp_perf, final_epsilon = trainer.train_model(
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

    # --------------------------
    # 5) Evaluate
    # --------------------------
    try:
        class_names = preprocessor.global_le.classes_.tolist()
        print(f"✅ Class names loaded: {class_names}")
    except AttributeError:
        # Fallback to hardcoded class names
        class_names = [
            'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
            'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
            'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
        ]
        print(f"⚠️ Using fallback class names: {len(class_names)} classes")

    eval_results = evaluate_model(model, X_test, y_test_cat, class_names=class_names)
    test_metrics = eval_results['test']
    print(f"✅ Evaluation complete:")
    print(f"   Test Loss: {test_metrics['loss']:.4f}")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test Macro F1: {test_metrics['macro_f1']:.4f}")

    # --------------------------
    # 6) Compress and upload weights
    # --------------------------
    print(f"\n{'='*70}")
    print("🗜️ Compressing model weights...")
    print(f"{'='*70}\n")
    
    # Choose compression method here:
    # Options: 'float16', 'tflite_dynamic', 'int8', 'none'
    compression_results = compress_and_save_weights(
        client_id,
        model,
        X_train,
        SAVE_DIR,
        compression_method='float16'  # RECOMMENDED for balance
    )
    
    # Prepare metadata for upload
    metadata = {
        "client_id": client_id,
        "final_test_loss": str(test_metrics['loss']),
        "final_test_accuracy": str(test_metrics['accuracy']),
        "final_test_precision": str(test_metrics['macro_precision']),
        "final_test_recall": str(test_metrics['macro_recall']),
        "final_test_f1": str(test_metrics['macro_f1']),
        "num_training_samples": str(num_samples),
        "compression_method": compression_results['primary']['method'],
        "compression_ratio": str(compression_results['primary']['compression_ratio']),
        "compressed_size_mb": str(compression_results['primary']['size_mb']),
        "original_size_mb": str(compression_results['primary']['original_size_mb'])
    }

    complete_metadata = {
        "test_metrics": test_metrics,
        "num_training_samples": str(num_samples),
        "data_classes_present": int(num_classes_train),
        "batch_size": config['batch_size'],
        "learning_rate": config['learning_rate'],
        "differential_privacy": True,
        "noise_multiplier": config['noise_multiplier'],
        "final_epsilon": final_epsilon,
        "delta": config['delta'],
        "compression": {
            "method": compression_results['primary']['method'],
            "ratio": compression_results['primary']['compression_ratio'],
            "original_size_mb": compression_results['primary']['original_size_mb'],
            "compressed_size_mb": compression_results['primary']['size_mb']
        }
    }

    # Upload compressed weights
    compressed_path = compression_results['primary']['path']
    upload_file(compressed_path, CLIENT_CONTAINER_NAME, metadata)
    
    # Upload complete metadata
    metadata_filename = get_versioned_metadata_filename(client_id, SAVE_DIR)
    uploaded_metadata = upload_json_to_blob(complete_metadata, metadata_filename, CLIENT_CONTAINER_NAME, {})

    print(f"\n✅ Client {client_id} completed with compression!")
    print(f"   Method: {compression_results['primary']['method']}")
    print(f"   Compression: {compression_results['primary']['compression_ratio']:.2f}x")
    print(f"   Size: {compression_results['primary']['original_size_mb']:.2f} MB → {compression_results['primary']['size_mb']:.2f} MB")
    
    return eval_results


if __name__ == "__main__":
    CLIENT_ID = os.getenv("CLIENT_ID")
    main(CLIENT_ID, epochs=2)
