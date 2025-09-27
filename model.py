import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2, l1_l2

class IoTModel:
    def __init__(self):
        """Initialize the IoT model class"""
        pass

    def create_mlp_model(self, input_dim, num_classes, architecture=[256, 256]):
        """
        Create FL-compatible MLP model matching successful notebook architecture
        
        Research-proven architecture: [256, 256] layers with 5e-5 learning rate
        FL Requirements:
        - No BatchNormalization (causes FL synchronization issues)
        - No Dropout (inconsistent between training/inference in FL)
        - Consistent layer naming for weight aggregation
        
        Returns:
        --------
        model : tf.keras.models.Sequential
            FL-compatible model with research-grade architecture
        """
        model = Sequential()
        
        # Build deep architecture with consistent naming (FL requirement)
        for i, units in enumerate(architecture):
            if i == 0:
                model.add(Dense(units, 
                               input_dim=input_dim, 
                               kernel_regularizer=l2(0.001), 
                               activation='relu', 
                               name=f'dense_{i}'))
            else:
                model.add(Dense(units, 
                               kernel_regularizer=l2(0.001), 
                               activation='relu', 
                               name=f'dense_{i}'))
        
        # Output layer
        model.add(Dense(num_classes, activation='softmax', name='output_dense'))
        
        return model

    def load_model(self, model_path):
        """
        Load a saved model from disk

        Parameters:
        -----------
        model_path : str
            Path to the saved model

        Returns:
        --------
        model : tf.keras.models.Sequential
            Loaded model
        """
        return tf.keras.models.load_model(model_path)

    def load_model_weights(self, model, weights_path):
        """
        Load model weights from disk

        Parameters:
        -----------
        model : tf.keras.models.Sequential
            Model to load weights into
        weights_path : str
            Path to the saved weights

        Returns:
        --------
        model : tf.keras.models.Sequential
            Model with loaded weights
        """
        model.load_weights(weights_path)
        return model 