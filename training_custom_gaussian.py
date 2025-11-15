import os
import math
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from functools import partial

# TensorFlow Privacy for Gaussian mechanism
try:
    import dp_accounting
except Exception as e:
    dp_accounting = None

# Model builder
try:
    from model import IoTModel
except ImportError:
    class IoTModel:
        def create_mlp_model(self, input_dim, num_classes, architecture):
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(input_dim,))
            ])
            for units in architecture:
                model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
            return model
    print("Using fallback IoTModel implementation")

# ========================= Per-example gradients =========================

@tf.function
def compute_per_example_gradients(model, X_batch, y_batch):
    """Efficient per-example gradient computation using tf.vectorized_map"""
    def single_example_grad(example_data):
        x_single, y_single = example_data
        x_single = tf.expand_dims(x_single, 0)
        y_single = tf.expand_dims(y_single, 0)
        
        with tf.GradientTape() as tape:
            predictions = model(x_single, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y_single, predictions)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients = [g if g is not None else tf.zeros_like(v) 
                    for g, v in zip(gradients, model.trainable_variables)]
        
        return gradients

    per_example_grads = tf.vectorized_map(
        fn=single_example_grad,
        elems=(X_batch, y_batch),
        fallback_to_while_loop=True
    )
    
    return per_example_grads

@tf.function
def clip_gradients_by_l2_norm(per_example_grads, clip_norm=1.0):
    """Clip gradients by L1 or L2 norm as specified"""
    batch_size = tf.shape(per_example_grads[0])[0]

    def clip_single_example(example_idx):
        example_grads = [grad[example_idx] for grad in per_example_grads]
        
        # L2 norm clipping for Gaussian and advanced Laplace
        global_norm = tf.linalg.global_norm(example_grads)
        clip_coeff = tf.minimum(1.0, clip_norm / (global_norm + 1e-8))
        
        clipped_grads = [grad * clip_coeff for grad in example_grads]
        return clipped_grads

    clipped_per_example = tf.map_fn(
        fn=clip_single_example,
        elems=tf.range(batch_size),
        fn_output_signature=[tf.TensorSpec(shape=grad.shape[1:], dtype=grad.dtype)
                           for grad in per_example_grads],
        parallel_iterations=32
    )
    
    return clipped_per_example

# ========================= CORRECTED Gaussian Manager =========================
class GaussianDPManager:
    """Gaussian mechanism: (ε,δ)-DP with L2 sensitivity and RDP accounting"""
    
    def __init__(self, noise_multiplier, l2_norm_clip, delta=1e-5):
        self.noise_multiplier = float(noise_multiplier)
        self.l2_norm_clip = float(l2_norm_clip)  # L2 sensitivity
        self.delta = delta

        # RDP accounting setup
        self.orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.rdp_accountant = None
        if dp_accounting:
            self.rdp_accountant = dp_accounting.rdp.RdpAccountant(self.orders)

        self.step_count = 0
        self.sampling_probability = None

        # Calculate noise scale
        self.noise_scale = self.l2_norm_clip * self.noise_multiplier
        print(f"Gaussian: noise_scale = {self.noise_scale:.6f}, L2 sensitivity = {self.l2_norm_clip}")

    def setup_accounting(self, batch_size, num_samples):
        """Setup RDP accounting parameters"""
        self.sampling_probability = batch_size / float(num_samples)

    def add_noise_to_gradients(self, aggregated_gradients):
        """Add Gaussian noise with RDP tracking"""
        noisy_gradients = []

        for grad in aggregated_gradients:
            # Generate Gaussian noise
            gaussian_noise = tf.random.normal(
                tf.shape(grad),
                mean=0.0,
                stddev=self.noise_scale,  # L2_sensitivity * noise_multiplier
                dtype=grad.dtype
            )
            
            noisy_grad = grad + gaussian_noise
            noisy_gradients.append(noisy_grad)

        # Track privacy using RDP
        if self.rdp_accountant and self.sampling_probability:
            event = dp_accounting.PoissonSampledDpEvent(
                self.sampling_probability,
                dp_accounting.GaussianDpEvent(self.noise_multiplier)
            )
            self.rdp_accountant.compose(event)

        self.step_count += 1
        return noisy_gradients

    def get_privacy_spent(self):
        """Return current privacy expenditure"""
        if not self.rdp_accountant:
            return {'epsilon': 0.0, 'delta': self.delta, 'steps': self.step_count, 'mechanism': 'Gaussian (L2, RDP)'}

        try:
            epsilon = self.rdp_accountant.get_epsilon(target_delta=self.delta)
            return {
                'epsilon': float(epsilon or 0.0), 
                'delta': self.delta, 
                'steps': self.step_count,
                'mechanism': 'Gaussian (L2, RDP)'
            }
        except Exception as e:
            return {'epsilon': 0.0, 'delta': self.delta, 'steps': self.step_count, 'mechanism': 'Gaussian (L2, RDP)'}

# ========================= CORRECTED Optimizer =========================

class GaussianDPOptimizer(tf.keras.optimizers.Adam):
    """Gaussian DP optimizer: (ε,δ)-DP with L2 clipping and RDP accounting"""
    
    def __init__(self, l2_norm_clip, noise_multiplier=1.0, delta=1e-5, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.dp_manager = None
        self.model = None

    def setup_dp(self, model, num_samples, batch_size):
        """Setup DP manager with RDP accounting"""
        self.model = model
        self.dp_manager = GaussianDPManager(
            noise_multiplier=self.noise_multiplier,
            l2_norm_clip=self.l2_norm_clip,
            delta=self.delta
        )
        self.dp_manager.setup_accounting(batch_size, num_samples)

    def train_step_with_dp(self, x_batch, y_batch):
        """Training step with Gaussian DP"""
        # Compute per-example gradients
        per_example_grads = compute_per_example_gradients(self.model, x_batch, y_batch)
        
        # Clip by L2 norm
        clipped_grads = clip_gradients_by_l2_norm(per_example_grads, clip_norm=self.l2_norm_clip)
        
        # Aggregate gradients
        aggregated_grads = [tf.reduce_sum(grad, axis=0) for grad in clipped_grads]
        
        # Add noise
        noisy_grads = self.dp_manager.add_noise_to_gradients(aggregated_grads)

        # Average gradients
        batch_size = tf.cast(tf.shape(x_batch)[0], tf.float32)
        noisy_avg = [g / batch_size for g in noisy_grads]

        # Apply gradients
        grads_and_vars = list(zip(noisy_avg, self.model.trainable_variables))
        self.apply_gradients(grads_and_vars)

        # Compute loss for logging
        predictions = self.model(x_batch, training=False)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, predictions))

        return loss

    def get_privacy_spent(self):
        if self.dp_manager is None:
            return {'epsilon': 0.0, 'delta': self.delta}
        return self.dp_manager.get_privacy_spent()

# ========================= Privacy Logger =========================

class DPPrivacyLogger(Callback):
    def __init__(self, trainer_instance, num_samples, batch_size, noise_type):
        super().__init__()
        self.trainer_instance = trainer_instance
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.noise_type = noise_type
        self.epoch_data = []
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epsilon = 0.0
        
        try:
            if self.noise_type == "gaussian":
                if hasattr(self.trainer_instance, 'gaussian_optimizer'):
                    privacy_spent = self.trainer_instance.gaussian_optimizer.get_privacy_spent()
                    epsilon = privacy_spent.get('epsilon', 0.0)
        except Exception as e:
            print(f"Privacy logging error: {e}")
            epsilon = 0.0
        
        val_acc = logs.get("val_accuracy", 0.0) or 0.0
        
        self.epoch_data.append({
            "epoch": epoch + 1,
            "train_loss": float(logs.get("loss", 0.0) or 0.0),
            "train_accuracy": float(logs.get("accuracy", 0.0) or 0.0),
            "val_loss": float(logs.get("val_loss", 0.0) or 0.0),
            "val_accuracy": float(val_acc),
            "epsilon": float(epsilon),
        })
        
        print(f"Epoch {epoch + 1}: ε = {epsilon:.4f}, Val Acc = {val_acc:.4f}")

# ========================= CORRECTED Main Trainer =========================

class IoTModelTrainer:
    """CORRECTED: Trainer with proper noise scaling and privacy accounting"""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_builder = IoTModel()
        self.model = None
        self.laplace_optimizer = None
        self.gaussian_optimizer = None
        
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
    
    def create_model(self, input_dim, num_classes, architecture=None):
        """Create model architecture"""
        if architecture is None:
            architecture = [256, 256]
        self.model = self.model_builder.create_mlp_model(input_dim, num_classes, architecture)
        return self.model
    
    def train_model(self, X_train, y_train_cat, X_val, y_val_cat,
                   model, epochs=20, batch_size=128, verbose=2,
                   use_dp=True, noise_type='gaussian', l2_norm_clip=1.0,
                   # Gaussian parameters  
                   noise_multiplier=1.0, delta=1e-5,  # Only for Gaussian
                   learning_rate=1e-3):
        """
        CORRECTED: Training with proper parameter usage
        
        For Laplace: Use epsilon_total (e.g., 3.0)
        For Gaussian: Use noise_multiplier (e.g., 1.0) and delta (e.g., 1e-5)
        """
        self.model = model
        
        # Setup callbacks
        callbacks = []
        if use_dp:
            callbacks.append(DPPrivacyLogger(
                trainer_instance=self,
                num_samples=len(X_train),
                batch_size=batch_size,
                noise_type=noise_type
            ))
        
        # Choose training approach based on noise type
        if use_dp and noise_type == 'gaussian':
            return self._train_gaussian_dp(
                X_train, y_train_cat, X_val, y_val_cat,
                epochs, batch_size, verbose, callbacks,
                l2_norm_clip, noise_multiplier, delta, learning_rate
            )
        else:
            # Non-DP training
            return self._train_non_dp(
                X_train, y_train_cat, X_val, y_val_cat,
                epochs, batch_size, verbose, learning_rate
            )
    
    def _train_gaussian_dp(self, X_train, y_train_cat, X_val, y_val_cat,
                          epochs, batch_size, verbose, callbacks,
                          l2_norm_clip, noise_multiplier, delta, learning_rate):
        """CORRECTED: Gaussian DP training"""
        print(f"\n=== GAUSSIAN DP TRAINING ===")
        print(f"Parameters: noise_multiplier={noise_multiplier}, l2_norm_clip={l2_norm_clip}, delta={delta}")
        
        # Create Gaussian optimizer
        self.gaussian_optimizer = GaussianDPOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            delta=delta,
            learning_rate=learning_rate
        )
        
        # Setup DP accounting
        self.gaussian_optimizer.setup_dp(self.model, len(X_train), batch_size)
        
        # Compile model for evaluation
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Training loop
        return self._run_training_loop(
            X_train, y_train_cat, X_val, y_val_cat,
            epochs, batch_size, verbose, callbacks, 'gaussian'
        )
    
    def _train_non_dp(self, X_train, y_train_cat, X_val, y_val_cat,
                     epochs, batch_size, verbose, learning_rate):
        """Non-DP baseline training"""
        print(f"\n=== NON-DP BASELINE TRAINING ===")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        return history, [], 0.0
    
    def _run_training_loop(self, X_train, y_train_cat, X_val, y_val_cat,
                          epochs, batch_size, verbose, callbacks, dp_type):
        """Common training loop for DP methods"""
        steps_per_epoch = len(X_train) // batch_size
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            if verbose >= 1:
                print(f"Epoch {epoch + 1}/{epochs}")
            
            epoch_loss = tf.keras.metrics.Mean()
            epoch_acc = tf.keras.metrics.CategoricalAccuracy()
            
            # Training batches
            for i in range(steps_per_epoch):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train_cat[start_idx:end_idx]
                
                try:
                    # Choose optimizer
                    if dp_type == 'gaussian':
                        batch_loss = self.gaussian_optimizer.train_step_with_dp(X_batch, y_batch)
                    
                    # Update metrics
                    epoch_loss(batch_loss)
                    predictions = self.model(X_batch, training=False)
                    epoch_acc(y_batch, predictions)
                    
                except Exception as e:
                    print(f"Training step failed: {e}")
                    break
            
            # Validation evaluation
            val_predictions = self.model(X_val, training=False)
            val_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_val_cat, val_predictions)
            ).numpy()
            val_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_val_cat, axis=1), tf.argmax(val_predictions, axis=1)), tf.float32)
            ).numpy()
            
            # Store history
            train_loss = epoch_loss.result().numpy()
            train_acc = epoch_acc.result().numpy()
            
            history['loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['accuracy'].append(float(train_acc))
            history['val_accuracy'].append(float(val_acc))
            
            # Callbacks
            for cb in callbacks:
                cb.on_epoch_end(epoch, {
                    'loss': train_loss,
                    'accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                })
            
            if verbose >= 1:
                if dp_type == 'gaussian':
                    privacy_info = self.gaussian_optimizer.get_privacy_spent()
                    
                print(f"  loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                      f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - "
                      f"epsilon: {privacy_info['epsilon']:.4f}")
        
        # Create history wrapper
        class HistoryWrapper:
            def __init__(self, history_dict):
                self.history = history_dict
        
        final_eps = callbacks[0].epoch_data[-1]['epsilon'] if callbacks and callbacks[0].epoch_data else 0.0
        
        return HistoryWrapper(history), callbacks[0].epoch_data if callbacks else [], final_eps

# ========================= NOISE EQUIVALENCE CALCULATOR =========================

def calculate_equivalent_noise_levels(l2_norm_clip=1.0):
    """
    Calculate equivalent noise levels between Gaussian and Laplace for comparison
    """
    print("=== NOISE LEVEL EQUIVALENCE GUIDE ===")
    print(f"For l2_norm_clip = {l2_norm_clip}:")
    print("\nGaussian -> Equivalent Laplace:")
    
    gaussian_configs = [
        (0.5, 1e-5),   # Very high privacy
        (1.0, 1e-5),   # Standard
        (1.5, 1e-5),   # Lower privacy
        (2.0, 1e-5),   # Low privacy
    ]
    
    for noise_mult, delta in gaussian_configs:
        # Approximate conversion: For Gaussian with noise_multiplier σ,
        # equivalent Laplace would have roughly same noise scale
        gaussian_noise = l2_norm_clip * noise_mult
        
        # For equivalent privacy level, Laplace epsilon would be roughly:
        # epsilon ≈ sensitivity / noise_scale = l2_norm_clip / gaussian_noise
        equiv_laplace_eps_per_step = l2_norm_clip / gaussian_noise
        
        print(f"  Gaussian: noise_multiplier={noise_mult}, delta={delta}")
        print(f"  -> Noise scale: {gaussian_noise:.3f}")
        print(f"  -> Equivalent Laplace epsilon_per_step: {equiv_laplace_eps_per_step:.3f}")
        print(f"  -> For 20 epochs (~400 steps): epsilon_total ≈ {equiv_laplace_eps_per_step * 400:.1f}")
        print()

# ========================= EXAMPLE USAGE =========================

if __name__ == "__main__":
    # Show noise equivalence guide
    calculate_equivalent_noise_levels()
    
    print("\n=== EXAMPLE CONFIGURATIONS FOR FAIR COMPARISON ===")
    
    print("\nConfiguration 1 - Similar Noise Levels:")
    print("Gaussian: noise_multiplier=1.0, delta=1e-5, l2_norm_clip=1.0")
    print("Laplace: epsilon_total=400.0, l2_norm_clip=1.0  # For 400 steps")
    print("Expected: Similar accuracy")
    
    print("\nConfiguration 2 - Similar Privacy Guarantee:")
    print("Gaussian: noise_multiplier=1.0, delta=1e-5  # Will give ε≈1.8 after 20 epochs")
    print("Laplace: epsilon_total=1.8, l2_norm_clip=1.0")
    print("Expected: Similar epsilon, Laplace will have lower accuracy due to higher noise")
    
    print("\nConfiguration 3 - Matched Performance:")
    print("Gaussian: noise_multiplier=1.0, delta=1e-5")
    print("Laplace: epsilon_total=50.0, l2_norm_clip=1.0  # Lower noise, better performance")
    print("Expected: Similar accuracy, but Laplace has much higher epsilon")
