import tensorflow as tf
import numpy as np
import os

class TFLiteCompressor:
    def __init__(self):
        """Initialize TFLite compressor for weight compression"""
        pass
    
    def compress_model(self, model, optimization_mode='default'):
        """
        Convert model to TFLite format with compression
        
        Parameters:
        -----------
        model : tf.keras.Model
            Trained model to compress
        optimization_mode : str
            'default', 'size', 'float16', or 'dynamic_range'
        
        Returns:
        --------
        tflite_model : bytes
            Compressed model in TFLite format
        compression_ratio : float
            Size reduction ratio
        """
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply optimization based on mode
        if optimization_mode == 'size':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        elif optimization_mode == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif optimization_mode == 'dynamic_range':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Calculate compression ratio
        original_size = self._get_model_size(model)
        compressed_size = len(tflite_model)
        compression_ratio = original_size / compressed_size
        
        return tflite_model, compression_ratio
    
    def _get_model_size(self, model):
        """Calculate model size in bytes"""
        temp_path = 'temp_model.h5'
        model.save(temp_path)
        size = os.path.getsize(temp_path)
        os.remove(temp_path)
        return size
    
    def save_compressed_model(self, tflite_model, save_path):
        """Save TFLite model to disk"""
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
    
    def load_compressed_model(self, model_path):
        """Load TFLite model from disk"""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def extract_weights_from_tflite(self, tflite_model_bytes):
        """
        Extract weights from TFLite model for federated aggregation
        
        Returns:
        --------
        weights : list
            List of weight arrays compatible with FL aggregation
        """
        # Create interpreter from bytes
        interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
        interpreter.allocate_tensors()
        
        # Extract weights
        weights = []
        for detail in interpreter.get_tensor_details():
            if 'weight' in detail['name'] or 'kernel' in detail['name']:
                tensor = interpreter.get_tensor(detail['index'])
                weights.append(tensor)
        
        return weights
