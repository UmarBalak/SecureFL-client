import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
import dp_accounting
import psutil
from model import IoTModel

class IoTModelTrainer:
    def __init__(self, random_state=42):
        """
        Initialize the model trainer

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.model_builder = IoTModel()
        self.model = None
        self.history = None

        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

    def create_model(self, input_dim, num_classes, architecture=None, dropout_rate_for_1=0.25, dropout_rate_for_all=0.2):
        """
        Create a new MLP model

        Parameters:
        -----------
        input_dim : int
            Number of input features
        num_classes : int
            Number of output classes
        architecture : list of int, optional
            List specifying the number of units in each hidden layer
        dropout_rate_for_1 : float
            Dropout rate for the first hidden layer
        dropout_rate_for_all : float
            Dropout rate for subsequent hidden layers

        Returns:
        --------
        model : tf.keras.models.Sequential
            Compiled MLP model
        """
        self.model = self.model_builder.create_mlp_model(input_dim, num_classes, architecture)
        return self.model

    def compute_epsilon(self, num_samples, batch_size, noise_multiplier, epochs_list, delta):
        """
        Compute epsilon for a list of epoch checkpoints.

        Parameters:
        -----------
        num_samples : int
            Number of training samples
        batch_size : int
            Batch size
        noise_multiplier : float
            Noise multiplier for DP-SGD
        epochs_list : list
            List of epoch checkpoints
        delta : float
            Delta value for DP

        Returns:
        --------
        epsilon_dict : dict
            Dictionary mapping epochs to epsilon values
        """
        epsilon_dict = {}
        if noise_multiplier == 0.0:
            return {e: float("inf") for e in epochs_list}
        
        sampling_probability = batch_size / num_samples
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        
        for e in epochs_list:
            accountant = dp_accounting.rdp.RdpAccountant(orders)
            steps = e * num_samples // batch_size
            event = dp_accounting.SelfComposedDpEvent(
                dp_accounting.PoissonSampledDpEvent(
                    sampling_probability,
                    dp_accounting.GaussianDpEvent(noise_multiplier)
                ),
                steps
            )
            accountant.compose(event)
            epsilon_dict[e] = accountant.get_epsilon(target_delta=delta)
        
        return epsilon_dict
    
    
    def train_model(self, X_train, y_train_cat, X_val, y_val_cat,
                model, architecture, epochs=50, batch_size=64, verbose=2,
                use_dp=True, l2_norm_clip=1.0, noise_multiplier=1.2, microbatches=1,
                callbacks=None, epoch_checkpoints=None, learning_rate=0.0005):
        """
        Train the MLP model with optional differential privacy.

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features.
        y_train_cat : numpy.ndarray
            One-hot encoded training labels.
        X_val : numpy.ndarray
            Validation features.
        y_val_cat : numpy.ndarray
            One-hot encoded validation labels.
        model : tf.keras.models.Sequential
            The model to train.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        verbose : int
            Verbosity level for training output.
        use_dp : bool
            Whether to use differential privacy.
        l2_norm_clip : float
            Clipping norm for DP-SGD.
        noise_multiplier : float
            Noise multiplier for DP-SGD.
        microbatches : int
            Number of microbatches for DP-SGD.
        callbacks : list
            List of Keras callbacks.
        epoch_checkpoints : list
            List of epoch checkpoints for epsilon and timing (e.g., [30, 50, 80, 100]).

        Returns:
        --------
        history : tf.keras.callbacks.History
            Training history.
        training_time : float
            Time taken for model.fit in seconds.
        noise_multiplier : float
            Noise multiplier used.
        l2_norm_clip : float
            L2 norm clip used.
        microbatches : int
        epsilon_dict : dict
            Epsilon values for each checkpoint epoch.
        delta : float
            Delta value used.
        epoch_times : list
            Cumulative time to each epoch end.
        memory_samples : list
            Memory usage (RSS in bytes) at the end of each epoch.
        """
        
        print("\nTraining MLP model...")
        self.model = model
        num_samples = len(X_train)
        epoch_checkpoints = epoch_checkpoints or [epochs]
        callbacks = callbacks or []

        if use_dp and DPKerasAdamOptimizer is None:
            print("DPKerasAdamOptimizer unavailable. Falling back to standard Adam.")
            use_dp = False

        if use_dp:
            if microbatches > batch_size:
                microbatches = batch_size
            for i in range(microbatches, 0, -1):
                if batch_size % i == 0:
                    microbatches = i
                    break
            print(f"Using {microbatches} microbatches with batch size {batch_size}")
            
            optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=learning_rate
            )
            
            delta = 1.0 / num_samples
            
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            delta = 1.0 / num_samples
            epsilon_dict = {e: float("inf") for e in epoch_checkpoints}
            print("Using standard Adam optimizer (no DP)")

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # wandb.init(
        #         project="securefl-iot",
        #         config={
        #             "epochs": epochs,
        #             "batch_size": batch_size,
        #             "architecture": architecture,
        #             "learning_rate": learning_rate,
        #             "use_dp": use_dp,
        #             "noise_multiplier": noise_multiplier,
        #             "l2_norm_clip": l2_norm_clip
        #         }
        #     )

        default_callbacks = [
            ModelCheckpoint(
                filepath='models/best_mlp_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=2
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=5,
                min_lr=1e-5,
                verbose=2
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=2
            )
        ]
        all_callbacks = default_callbacks + callbacks
        # all_callbacks = default_callbacks + callbacks + [
        #     WandbMetricsLogger(log_freq=5),
        #     WandbModelCheckpoint("models")   # will push best checkpoints to W&B
        # ]


        start_time = time.time()
        mem_start = psutil.Process().memory_info().rss
        try:

            self.history = self.model.fit(
                X_train, y_train_cat,
                validation_data=(X_val, y_val_cat),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=all_callbacks,
                verbose=verbose
            )
            training_success = True
        except Exception as e:
            print(f"Training failed: {str(e)}")
            print("Trying simpler DP configuration...")
            
            if use_dp:
                optimizer = DPKerasAdamOptimizer(
                    l2_norm_clip=l2_norm_clip,
                    noise_multiplier=noise_multiplier,
                    num_microbatches=1,
                    learning_rate=learning_rate
                )
                self.model.compile(
                    optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                try:

                    self.history = self.model.fit(
                        X_train, y_train_cat,
                        validation_data=(X_val, y_val_cat),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=all_callbacks,
                        verbose=verbose,

                    )
                    training_success = True
                except Exception as e2:
                    print(f"Training failed again: {str(e2)}")
                    print("Falling back to non-DP optimizer")
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    self.model.compile(
                        optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    self.history = self.model.fit(
                        X_train, y_train_cat,
                        validation_data=(X_val, y_val_cat),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=all_callbacks,
                        verbose=verbose
                    )
                    training_success = True
        
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        # wandb.finish()

        if use_dp:
            actual_epochs = len(self.history.history['loss'])
            epsilon_dict = self.compute_epsilon(num_samples, batch_size, noise_multiplier, [actual_epochs], delta)
            
            for e, eps in epsilon_dict.items():
                print(f"DP guarantee at epoch {e}: ε = {eps:.2f}, δ = {delta:.2e}")
        

        return self.history, training_time, num_samples, delta, epsilon_dict

    
    def get_model(self):
        """
        Get the trained model

        Returns:
        --------
        model : tf.keras.models.Sequential
            Trained model
        """
        return self.model

    def get_history(self):
        """
        Get the training history

        Returns:
        --------
        history : tf.keras.callbacks.History
            Training history
        """
        return self.history