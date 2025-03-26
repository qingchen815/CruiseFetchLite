import tensorflow as tf
import numpy as np

class TLITE(tf.keras.Model):
    '''
    T-LITE: A lightweight version of Twilight neural prefetcher
    with behavioral clustering and frequency-based candidate selection
    '''

    def __init__(self, config):
        super(TLITE, self).__init__()
        
        # Model hyperparameters
        self.pc_embed_size = config.pc_embed_size            # 64
        self.cluster_embed_size = config.cluster_embed_size  # 25
        self.offset_embed_size = config.offset_embed_size    # 2500 (cluster_embed_size * num_experts)
        self.num_experts = config.num_experts                # 100
        self.history_length = config.history_length          # 3
        self.num_pcs = config.num_pcs                        # 4096 max
        self.num_clusters = config.num_clusters              # 4096
        self.offset_size = config.offset_size                # 64 (typically 2^offset_bits)
        self.num_candidates = config.num_candidates          # 4
        self.dpf_history_length = config.dpf_history_length  # 1
        self.steps_per_epoch = config.steps_per_epoch        # For tracking training state
        
        # Static variables to track internal state
        self.step = 0
        self.epoch = 0
        
        # Initialize model components
        self.init_embedding_layers()
        self.init_prediction_layers()
    
    def init_embedding_layers(self):
        '''Initialize embedding layers for PC, cluster, and offset'''
        # PC embedding layer (limited to top N most common PCs)
        self.pc_embedding = tf.keras.layers.Embedding(
            self.num_pcs, 
            self.pc_embed_size,
            embeddings_regularizer='l1'
        )
        
        # Cluster embedding layer 
        self.cluster_embedding = tf.keras.layers.Embedding(
            self.num_clusters,
            self.cluster_embed_size,
            embeddings_regularizer='l1'
        )
        
        # Offset embedding layer divided into experts
        self.offset_embedding = tf.keras.layers.Embedding(
            self.offset_size,
            self.offset_embed_size,
            embeddings_regularizer='l1'
        )
        
        # Multi-head attention for context-aware offset embedding
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=1,
            key_dim=self.cluster_embed_size,
            attention_axes=(2, 3),
            kernel_regularizer='l1',
        )
    
    def init_prediction_layers(self):
        '''Initialize fully connected layers for candidate and offset prediction'''
        # Fully connected layer for candidate prediction (N+1 outputs for N candidates + no prefetch)
        self.candidate_fc = tf.keras.layers.Dense(
            self.num_candidates + 1,
            activation=None,
            kernel_regularizer='l1'
        )
        
        # Fully connected layer for offset prediction
        self.offset_fc = tf.keras.layers.Dense(
            self.offset_size,
            activation=None,
            kernel_regularizer='l1'
        )
    
    def compute_context_aware_offset_embedding(self, cluster_history, offset_history, pc, training=False):
        '''
        Compute context-aware offset embeddings by incorporating cluster and PC context
        using the mixture-of-experts approach
        '''
        # Get raw embeddings
        cluster_embed = self.cluster_embedding(cluster_history)  # [batch, history, embed_dim]
        offset_embed = self.offset_embedding(offset_history)     # [batch, history, offset_embed_size]
        pc_embed = self.pc_embedding(pc)                        # [batch, pc_embed_size]
        
        # Reshape offset embeddings for attention mechanism
        # Split offset embedding into experts
        batch_size = tf.shape(offset_history)[0]
        offset_embed = tf.reshape(
            offset_embed, 
            shape=[-1, self.history_length, self.num_experts, self.cluster_embed_size]
        )
        
        # Create context embedding by concatenating cluster and PC
        # Expand dimensions for attention mechanism
        context_embed = tf.concat([
            tf.reshape(cluster_embed, [-1, self.history_length, 1, self.cluster_embed_size]),
            tf.reshape(pc_embed, [-1, 1, 1, self.pc_embed_size])
        ], axis=2)
        context_embed = tf.reshape(context_embed, [-1, self.history_length, 1, self.cluster_embed_size])
        
        # Apply attention to get weighted offset embeddings
        context_aware_offset = self.mha(
            context_embed,  # Query
            offset_embed,   # Key/Value
            training=training
        )
        
        # Reshape to final embedding dimension
        context_aware_offset = tf.reshape(
            context_aware_offset, 
            [-1, self.history_length, self.cluster_embed_size]
        )
        
        return context_aware_offset
    
    def call(self, inputs, training=False):
        '''
        Forward pass of the T-LITE model
        
        Inputs:
        - cluster_history: History of cluster IDs [batch, history_length]
        - offset_history: History of offsets [batch, history_length]
        - pc: Load PC [batch, 1]
        - dpf_vectors: DPF distribution vectors [batch, dpf_history_length, num_candidates]
        
        Returns:
        - candidate_logits: Logits for candidate prediction [batch, num_candidates+1]
        - offset_logits: Logits for offset prediction [batch, offset_size]
        '''
        # Unpack inputs
        cluster_history, offset_history, pc, dpf_vectors = inputs
        
        # Compute embeddings
        cluster_embed = self.cluster_embedding(cluster_history)  # [batch, history, embed_dim]
        context_aware_offset = self.compute_context_aware_offset_embedding(
            cluster_history, offset_history, pc, training
        )
        pc_embed = self.pc_embedding(pc)                        # [batch, pc_embed_dim]
        
        # Flatten DPF vectors
        dpf_flat = tf.reshape(
            dpf_vectors, 
            [-1, self.dpf_history_length * self.num_candidates]
        )
        
        # Flatten embeddings
        cluster_flat = tf.reshape(cluster_embed, [-1, self.history_length * self.cluster_embed_size])
        offset_flat = tf.reshape(context_aware_offset, [-1, self.history_length * self.cluster_embed_size])
        
        # Concatenate all features
        combined_features = tf.concat([
            pc_embed,           # PC embedding
            cluster_flat,       # Cluster history embeddings
            offset_flat,        # Context-aware offset embeddings
            dpf_flat            # DPF distribution vectors
        ], axis=1)
        
        # Make predictions
        candidate_logits = self.candidate_fc(combined_features)
        offset_logits = self.offset_fc(combined_features)
        
        return candidate_logits, offset_logits
    
    def train_step(self, data):
        '''Custom training step to update internal counters'''
        # Unpack data
        x, y = data
        
        # Forward pass
        with tf.GradientTape() as tape:
            candidate_logits, offset_logits = self(x, training=True)
            
            # Unpack ground truth
            candidate_labels, offset_labels = y
            
            # Compute losses
            candidate_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )(candidate_labels, candidate_logits)
            
            offset_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )(offset_labels, offset_logits)
            
            # Total loss
            total_loss = candidate_loss + offset_loss
        
        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update internal counters
        self.step += 1
        if self.step >= self.steps_per_epoch:
            self.step = 0
            self.epoch += 1
        
        # Update metrics
        self.compiled_metrics.update_state(y, (candidate_logits, offset_logits))
        
        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = total_loss
        
        return results
    
    def load(self, model_path):
        '''Load model weights'''
        self.load_weights(model_path).expect_partial()
    
    def save(self, model_path):
        '''Save model weights'''
        self.save_weights(model_path)
    
    def quantize(self, bits=8):
        '''
        Quantize model weights to specified bit precision
        
        This is a simple implementation - production would use TF's quantization API
        '''
        # Get all weights
        weights = self.get_weights()
        quantized_weights = []
        
        for w in weights:
            # Compute scale factor based on min/max
            w_min = np.min(w)
            w_max = np.max(w)
            scale = (w_max - w_min) / (2**bits - 1)
            
            # Quantize
            w_quant = np.round((w - w_min) / scale).astype(np.int8)
            
            # Store quantized weights
            quantized_weights.append(w_quant)
        
        # Set quantized weights
        self.set_weights(quantized_weights)
        
        return {
            'bits': bits,
            'orig_size_mb': sum(w.size * 4 for w in weights) / (1024 * 1024),
            'quant_size_mb': sum(w.size * bits / 8 for w in weights) / (1024 * 1024)
        }


def create_tlite_model(config):
    '''
    Create and compile the T-LITE model
    
    Args:
        config: Configuration object containing model hyperparameters
        
    Returns:
        Compiled T-LITE model
    '''
    # Create model
    model = TLITE(config)
    
    # Define a dummy input to build the model
    dummy_cluster_history = tf.zeros((1, model.history_length), dtype=tf.int32)
    dummy_offset_history = tf.zeros((1, model.history_length), dtype=tf.int32)
    dummy_pc = tf.zeros((1, 1), dtype=tf.int32)
    dummy_dpf = tf.zeros((1, model.dpf_history_length, model.num_candidates), dtype=tf.float32)
    
    # Build the model
    model((dummy_cluster_history, dummy_offset_history, dummy_pc, dummy_dpf))
    
    # Import metrics
    from metrics import CandidateAccuracy, OffsetAccuracy, OverallAccuracy
    
    # Define metrics
    metrics = [
        CandidateAccuracy(), 
        OffsetAccuracy(),
        OverallAccuracy()
    ]
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss=None,  # Loss is computed in train_step
        metrics=metrics
    )
    
    return model
