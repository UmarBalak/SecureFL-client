import tensorflow as tf
import numpy as np
import os

class QuantizationCompressor:
    def __init__(self):
        """Initialize post-training quantization compressor"""
        pass
    
    def compress_model(self, model, quantization_type='float16'):
        """
        Apply post-training quantization to model
        
        Parameters:
        -----------
        model : tf.keras.Model
            Trained model to compress
        quantization_type : str
            'float16'
        
        Returns:
        --------
        quantized_model : bytes
            Quantized TFLite model
        compression_ratio : float
            Size reduction ratio
        """
        # Get original size
        original_size = self._get_model_size(model)
        
        # Convert to TFLite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
        if quantization_type == 'float16':
            # Float16 quantization (2x size reduction)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        quantized_model = converter.convert()
        
        # Calculate compression ratio
        compressed_size = len(quantized_model)
        compression_ratio = original_size / compressed_size
        
        return quantized_model, compression_ratio
    
    def _get_model_size(self, model):
        """Calculate model size in bytes"""
        temp_path = 'temp_model.h5'
        model.save(temp_path)
        size = os.path.getsize(temp_path)
        os.remove(temp_path)
        return size
    
    def save_quantized_model(self, quantized_model, save_path):
        """Save quantized model to disk"""
        with open(save_path, 'wb') as f:
            f.write(quantized_model)
    
    def load_quantized_model(self, model_path):
        """Load quantized model from disk"""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
