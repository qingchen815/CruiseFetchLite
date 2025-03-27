import tensorflow as tf
import numpy as np
import sys  # For debug output

class TLITE(tf.keras.Model):
    '''
    T-LITE: A lightweight version of Twilight neural prefetcher
    with behavioral clustering and frequency-based candidate selection
    '''

    def __init__(self, config):
        super(TLITE, self).__init__()
        
        # Save configuration parameters
        self.pc_embed_size = config.pc_embed_size           
        self.cluster_embed_size = config.cluster_embed_size  
        self.offset_embed_size = config.offset_embed_size    
        self.num_experts = config.num_experts                
        self.history_length = config.history_length          
        self.num_pcs = config.num_pcs                        
        self.num_clusters = config.num_clusters              
        self.offset_size = config.offset_size                
        self.num_candidates = config.num_candidates          
        self.dpf_history_length = config.dpf_history_length  
        self.steps_per_epoch = config.steps_per_epoch        
        
        # Variables to track training state
        self.step = 0
        self.epoch = 0
        
        # Initialize model components
        self._init_embedding_layers()
        self._init_prediction_layers()
    
    def _init_embedding_layers(self):
        '''Initialize embedding layers'''
        # PC embedding layer (limited to N most common PCs)
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
        
        # Offset embedding layer (divided into multiple experts)
        self.offset_embedding = tf.keras.layers.Embedding(
            self.offset_size,
            self.offset_embed_size,
            embeddings_regularizer='l1'
        )
        
        # Multi-head attention for context-aware offset embedding
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=1,
            key_dim=self.cluster_embed_size,
            kernel_regularizer='l1'
        )
    
    def _init_prediction_layers(self):
        '''Initialize prediction layers'''
        # Candidate prediction layer (N+1 outputs for N candidates + no-prefetch option)
        self.candidate_fc = tf.keras.layers.Dense(
            self.num_candidates + 1,
            activation=None,
            kernel_regularizer='l1'
        )
        
        # Offset prediction layer
        self.offset_fc = tf.keras.layers.Dense(
            self.offset_size,
            activation=None,
            kernel_regularizer='l1'
        )
    
    def compute_context_aware_offset_embedding(self, cluster_history, offset_history, pc, training=False):
        '''
        Compute context-aware offset embedding
        Fuses clustering and PC context information using expert mixing method
        
        Args:
            cluster_history: Cluster ID history [batch, history_length]
            offset_history: Offset history [batch, history_length]
            pc: Load PC [batch, 1]
            training: Whether in training mode
            
        Returns:
            Context-aware offset embedding [batch, history_length, cluster_embed_size]
        '''
        # Get batch size
        batch_size = tf.shape(cluster_history)[0]
        
        # Get original embeddings
        cluster_embed = self.cluster_embedding(cluster_history)  # [batch, history, embed_dim]
        offset_embed = self.offset_embedding(offset_history)     # [batch, history, offset_embed_size]
        
        # Print shapes for debugging
        # tf.print("batch_size:", batch_size, "history_length:", self.history_length, 
        #          "cluster_embed shape:", tf.shape(cluster_embed),
        #          "offset_embed shape:", tf.shape(offset_embed),
        #          output_stream=sys.stderr)
        
        # Reshape offset embedding for expert format
        offset_embed = tf.reshape(
            offset_embed, 
            [batch_size, self.history_length, self.num_experts, self.cluster_embed_size]
        )
        
        # Prepare query - using only cluster embedding as query
        # This simplified method avoids dimension mismatch issues from past concatenation operations
        query = tf.reshape(
            cluster_embed, 
            [batch_size, self.history_length, 1, self.cluster_embed_size]
        )
        
        # Apply attention mechanism to get weighted offset embedding
        # Print query and value shapes for debugging
        # tf.print("query shape:", tf.shape(query), "value shape:", tf.shape(offset_embed), output_stream=sys.stderr)
        
        context_aware_offset = self.mha(
            query=query,        # [batch, history, 1, cluster_embed_size]
            value=offset_embed, # [batch, history, num_experts, cluster_embed_size]
            training=training
        )
        
        # Reshape to final embedding dimensions
        context_aware_offset = tf.reshape(
            context_aware_offset, 
            [batch_size, self.history_length, self.cluster_embed_size]
        )
        
        return context_aware_offset
    
    def call(self, inputs, training=False):
        '''
        Model forward pass
        
        Args:
            inputs: Tuple (cluster_history, offset_history, pc, dpf_vectors)
                - cluster_history: Cluster ID history [batch, history_length]
                - offset_history: Offset history [batch, history_length]
                - pc: Load PC [batch, 1]
                - dpf_vectors: DPF distribution vectors [batch, dpf_history_length, num_candidates]
            training: Whether in training mode
            
        Returns:
            candidate_logits: Candidate prediction logits [batch, num_candidates+1]
            offset_logits: Offset prediction logits [batch, offset_size]
        '''
        # Unpack inputs
        cluster_history, offset_history, pc, dpf_vectors = inputs
        
        # Get batch size for consistent dimension handling
        batch_size = tf.shape(cluster_history)[0]
        
        # Calculate embeddings
        cluster_embed = self.cluster_embedding(cluster_history)  # [batch, history, embed_dim]
        context_aware_offset = self.compute_context_aware_offset_embedding(
            cluster_history, offset_history, pc, training
        )
        pc_embed = self.pc_embedding(pc)  # [batch, 1, pc_embed_dim]
        
        # Ensure pc_embed is 2D - fix key point
        pc_embed = tf.reshape(pc_embed, [batch_size, self.pc_embed_size])
        
        # Flatten DPF vectors
        dpf_flat = tf.reshape(
            dpf_vectors, 
            [batch_size, self.dpf_history_length * self.num_candidates]
        )
        
        # Flatten embeddings
        cluster_flat = tf.reshape(cluster_embed, [batch_size, self.history_length * self.cluster_embed_size])
        offset_flat = tf.reshape(context_aware_offset, [batch_size, self.history_length * self.cluster_embed_size])
        
        # Concatenate all features
        combined_features = tf.concat([
            pc_embed,     # PC embedding
            cluster_flat, # Cluster history embedding
            offset_flat,  # Context-aware offset embedding
            dpf_flat      # DPF distribution vectors
        ], axis=1)
        
        # Generate predictions
        candidate_logits = self.candidate_fc(combined_features)
        offset_logits = self.offset_fc(combined_features)
        
        return candidate_logits, offset_logits
    
    def train_step(self, data):
        '''Custom training step, updates internal counters'''
        # Unpack data
        x, y = data
        
        # Forward pass
        with tf.GradientTape() as tape:
            candidate_logits, offset_logits = self(x, training=True)
            
            # Unpack ground truth labels
            candidate_labels, offset_labels = y
            
            # Calculate losses
            candidate_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )(candidate_labels, candidate_logits)
            
            offset_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )(offset_labels, offset_logits)
            
            # Total loss
            total_loss = candidate_loss + offset_loss
        
        # Calculate gradients and update weights
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
        
        This is a simple implementation - production should use TF's quantization API
        
        Args:
            bits: Quantization precision in bits
            
        Returns:
            Dictionary containing quantization information
        '''
        # Get all weights
        weights = self.get_weights()
        quantized_weights = []
        
        for w in weights:
            # Calculate scaling factor based on min/max values
            w_min = np.min(w)
            w_max = np.max(w)
            scale = (w_max - w_min) / (2**bits - 1)
            
            # Quantize
            if scale > 0:  # Prevent division by zero
                w_quant = np.round((w - w_min) / scale).astype(np.int8)
            else:
                w_quant = np.zeros_like(w, dtype=np.int8)
            
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
    Create and compile T-LITE model
    
    Args:
        config: Configuration object containing model hyperparameters
        
    Returns:
        Compiled T-LITE model
    '''
    # Create model
    model = TLITE(config)
    
    # Use small batch to test model structure
    batch_size = 2  # Use batch size >1 to test batch processing logic
    
    # Define test inputs to build model
    dummy_cluster_history = tf.zeros((batch_size, model.history_length), dtype=tf.int32)
    dummy_offset_history = tf.zeros((batch_size, model.history_length), dtype=tf.int32)
    dummy_pc = tf.zeros((batch_size, 1), dtype=tf.int32)
    dummy_dpf = tf.zeros((batch_size, model.dpf_history_length, model.num_candidates), dtype=tf.float32)
    
    # Build model
    # Uncomment following line for debugging
    # tf.print("Testing model with batch size:", batch_size, output_stream=sys.stderr)
    model((dummy_cluster_history, dummy_offset_history, dummy_pc, dummy_dpf))
    
    # Import metrics
    try:
        from metrics import CandidateAccuracy, OffsetAccuracy, OverallAccuracy
        
        # Define metrics
        metrics = [
            CandidateAccuracy(), 
            OffsetAccuracy(),
            OverallAccuracy()
        ]
    except ImportError:
        print("Warning: Could not import metrics module, using default metrics")
        metrics = ['accuracy']
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss=None,  # Loss is calculated in train_step
        metrics=metrics
    )
    
    return model