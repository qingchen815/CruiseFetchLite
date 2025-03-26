import numpy as np

class BehavioralClusteringUtils:
    '''
    Utilities for behavioral clustering of pages
    '''
    
    @staticmethod
    def extract_page_embeddings(model):
        '''
        Extract page embeddings from a trained Twilight model
        
        Args:
            model: Trained Twilight model with page embeddings
            
        Returns:
            Page embeddings as numpy array
        '''
        # Extract page embedding weights
        return model.page_embedding.get_weights()[0]
    
    @staticmethod
    def cluster_pages(page_embeddings, num_clusters=4096):
        '''
        Cluster page embeddings using k-means
        
        Args:
            page_embeddings: Page embeddings from a trained model
            num_clusters: Number of clusters to create
            
        Returns:
            cluster_map: Dictionary mapping page IDs to cluster IDs
            centroids: Cluster centroids
        '''
        from sklearn.cluster import KMeans
        
        # Create and fit KMeans model
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(page_embeddings)
        
        # Create mapping from page ID to cluster ID
        cluster_map = {page_id: cluster_id for page_id, cluster_id in enumerate(clusters)}
        
        return cluster_map, kmeans.cluster_centers_
    
    @staticmethod
    def create_tlite_from_twilight(twilight_model, config, cluster_map, centroids):
        '''
        Create and initialize a T-LITE model from a trained Twilight model
        using behavioral clustering
        
        Args:
            twilight_model: Trained Twilight model
            config: Configuration for T-LITE
            cluster_map: Mapping from page IDs to cluster IDs
            centroids: Cluster centroids
            
        Returns:
            Initialized T-LITE model
        '''
        from model import create_tlite_model
        
        # Create T-LITE model
        tlite_model = create_tlite_model(config)
        
        # Initialize cluster embeddings with centroids
        tlite_model.cluster_embedding.set_weights([centroids])
        
        # Transfer weights for other components
        # PC embedding
        tlite_model.pc_embedding.set_weights(twilight_model.pc_embedding.get_weights())
        
        # Offset embedding
        tlite_model.offset_embedding.set_weights(twilight_model.offset_embedding.get_weights())
        
        # MHA layer weights
        tlite_model.mha.set_weights(twilight_model.mha.get_weights())
        
        # Initialize prediction layers with Twilight's weights
        # This assumes compatible layer shapes - adjust as needed
        tlite_model.candidate_fc.set_weights([
            twilight_model.page_linear.get_weights()[0][:, :config.num_candidates+1],
            twilight_model.page_linear.get_weights()[1][:config.num_candidates+1]
        ])
        
        tlite_model.offset_fc.set_weights(twilight_model.offset_linear.get_weights())
        
        return tlite_model
    
    @staticmethod
    def dynamic_page_to_cluster_mapping(offset_transitions, cluster_offset_transitions):
        '''
        Dynamically map a page to a cluster based on its offset transition pattern
        
        Args:
            offset_transitions: Offset transition matrix for the page [64, 64]
            cluster_offset_transitions: List of offset transition matrices for each cluster
            
        Returns:
            Best matching cluster ID
        '''
        # If no transitions recorded yet, return a default cluster
        if np.sum(offset_transitions) == 0:
            return 0
        
        # If no cluster offset transitions recorded yet, return default
        if len(cluster_offset_transitions) == 0:
            return 0
        
        # Normalize the offset transitions
        norm_transitions = offset_transitions / (np.sum(offset_transitions) + 1e-10)
        
        # Calculate distance to each cluster
        distances = []
        for cluster_id, cluster_transitions in enumerate(cluster_offset_transitions):
            # Skip empty clusters
            if np.sum(cluster_transitions) == 0:
                distances.append(float('inf'))
                continue
                
            # Normalize cluster transitions
            norm_cluster = cluster_transitions / (np.sum(cluster_transitions) + 1e-10)
            
            # Calculate distance (using Frobenius norm)
            dist = np.linalg.norm(norm_transitions - norm_cluster)
            distances.append(dist)
        
        # Return cluster with minimum distance
        return np.argmin(distances)
    
    @staticmethod
    def aggregate_cluster_offset_transitions(cluster_map, page_metadata):
        '''
        Aggregate offset transitions for each cluster
        
        Args:
            cluster_map: Mapping from page IDs to cluster IDs
            page_metadata: Dictionary of page metadata
            
        Returns:
            List of aggregated offset transition matrices for each cluster
        '''
        # Find number of clusters
        if not cluster_map:
            return []
            
        num_clusters = max(cluster_map.values()) + 1
        
        # Initialize aggregated transitions
        cluster_transitions = [np.zeros((64, 64), dtype=np.float32) for _ in range(num_clusters)]
        
        # Aggregate transitions
        for page_id, metadata in page_metadata.items():
            if page_id in cluster_map:
                cluster_id = cluster_map[page_id]
                cluster_transitions[cluster_id] += metadata['offset_transitions']
        
        return cluster_transitions
    
    @staticmethod
    def visualize_clusters(page_embeddings, cluster_labels, n_samples=1000):
        '''
        Visualize clusters using t-SNE
        
        Args:
            page_embeddings: Page embeddings
            cluster_labels: Cluster labels for each page
            n_samples: Number of samples to visualize
            
        Returns:
            matplotlib figure
        '''
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            
            # Subsample if too many points
            if len(page_embeddings) > n_samples:
                indices = np.random.choice(len(page_embeddings), n_samples, replace=False)
                embeddings_sample = page_embeddings[indices]
                labels_sample = [cluster_labels[i] for i in indices]
            else:
                embeddings_sample = page_embeddings
                labels_sample = cluster_labels
            
            # Apply t-SNE dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42)
            reduced_data = tsne.fit_transform(embeddings_sample)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot points colored by cluster
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_sample, cmap='tab20', alpha=0.7)
            
            # Add legend and labels
            plt.colorbar(scatter, label='Cluster')
            plt.title('t-SNE Visualization of Page Embeddings')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            
            return plt.gcf()
        except ImportError:
            print("Visualization requires matplotlib and scikit-learn.")
            return None



def fine_tune_tlite(tlite_model, traces, cluster_map, config, epochs=10):
    '''
    Fine-tune a T-LITE model on memory traces
    
    Args:
        tlite_model: T-LITE model to fine-tune
        traces: Memory traces to use for fine-tuning
        cluster_map: Mapping from page IDs to cluster IDs
        config: Model configuration
        epochs: Number of epochs for fine-tuning
        
    Returns:
        Fine-tuned T-LITE model
    '''
    import tensorflow as tf
    from tqdm import tqdm
    
    # Configure fine-tuning
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    tlite_model.compile(optimizer=optimizer)
    
    # Create dataset from traces
    def create_dataset(traces):
        X = []
        y_candidates = []
        y_offsets = []
        
        for trace in tqdm(traces, desc="Processing traces"):
            # Extract page, offset, and PC sequences
            page_history = trace['page_history']
            offset_history = trace['offset_history']
            pc_history = trace['pc_history']
            next_page = trace['next_page']
            next_offset = trace['next_offset']
            
            # Convert pages to clusters
            cluster_history = [cluster_map.get(page, 0) for page in page_history]
            
            # Create DPF vector (dummy for fine-tuning)
            dpf_vector = np.zeros(config.num_candidates, dtype=np.float32)
            
            # Find the candidate index
            candidate_idx = config.num_candidates  # Default to no-prefetch
            for i in range(min(config.num_candidates, len(trace['candidates']))):
                if trace['candidates'][i][0] == next_page:
                    candidate_idx = i
                    break
            
            # Add to dataset
            X.append([
                np.array(cluster_history, dtype=np.int32),
                np.array(offset_history, dtype=np.int32),
                np.array([pc_history[-1]], dtype=np.int32),
                np.array([dpf_vector], dtype=np.float32)
            ])
            y_candidates.append(candidate_idx)
            y_offsets.append(next_offset)
        
        # Convert to TensorFlow dataset
        def generator():
            for i in range(len(X)):
                yield X[i], (y_candidates[i], y_offsets[i])
        
        # Create and batch dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(len(page_history),), dtype=tf.int32),
                    tf.TensorSpec(shape=(len(offset_history),), dtype=tf.int32),
                    tf.TensorSpec(shape=(1,), dtype=tf.int32),
                    tf.TensorSpec(shape=(1, config.num_candidates), dtype=tf.float32)
                ),
                (
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32)
                )
            )
        )
        
        return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create dataset
    train_dataset = create_dataset(traces)
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Fine-tune model
    tlite_model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return tlite_model


def convert_voyager_to_twilight(voyager_model, config):
    '''
    Convert a Voyager model to a Twilight model by incorporating
    frequency-based candidate selection
    
    Args:
        voyager_model: Voyager model
        config: Configuration for Twilight
        
    Returns:
        Twilight model
    '''
    # This is a placeholder implementation - the actual implementation would
    # depend on the specifics of the Voyager model architecture
    
    # Import needed modules
    import tensorflow as tf
    
    # Create Twilight model (similar structure to T-LITE but without behavioral clustering)
    # First extract embeddings and weights from Voyager
    pc_embeddings = voyager_model.pc_embedding.get_weights()
    page_embeddings = voyager_model.page_embedding.get_weights()
    offset_embeddings = voyager_model.offset_embedding.get_weights()
    mha_weights = voyager_model.mha.get_weights()
    
    # Create basic structure of Twilight model
    class Twilight(tf.keras.Model):
        def __init__(self, config):
            super(Twilight, self).__init__()
            
            # Model hyperparameters
            self.pc_embed_size = config.pc_embed_size            
            self.page_embed_size = config.page_embed_size  
            self.offset_embed_size = config.offset_embed_size    
            self.num_experts = config.num_experts                
            self.history_length = config.history_length          
            self.num_pcs = config.num_pcs                        
            self.num_pages = config.num_pages              
            self.offset_size = config.offset_size                
            self.num_candidates = config.num_candidates          
            self.dpf_history_length = config.dpf_history_length  
            
            # Embedding layers
            self.pc_embedding = tf.keras.layers.Embedding(
                self.num_pcs, 
                self.pc_embed_size,
                embeddings_regularizer='l1'
            )
            
            self.page_embedding = tf.keras.layers.Embedding(
                self.num_pages,
                self.page_embed_size,
                embeddings_regularizer='l1'
            )
            
            self.offset_embedding = tf.keras.layers.Embedding(
                self.offset_size,
                self.offset_embed_size,
                embeddings_regularizer='l1'
            )
            
            # MHA for context-aware offset embedding
            self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads=1,
                key_dim=self.page_embed_size,
                attention_axes=(2, 3),
                kernel_regularizer='l1',
            )
            
            # Prediction layers
            self.candidate_fc = tf.keras.layers.Dense(
                self.num_candidates + 1,
                activation=None,
                kernel_regularizer='l1'
            )
            
            self.offset_fc = tf.keras.layers.Dense(
                self.offset_size,
                activation=None,
                kernel_regularizer='l1'
            )
            
        def call(self, inputs, training=False):
            # Implement similar to TLITE but with pages instead of clusters
            page_history, offset_history, pc, dpf_vectors = inputs
            
            # Compute embeddings
            page_embed = self.page_embedding(page_history)
            offset_embed = self.offset_embedding(offset_history)
            pc_embed = self.pc_embedding(pc)
            
            # Context-aware offset embedding
            # Implementation similar to TLITE.compute_context_aware_offset_embedding
            
            # Concatenate features and make predictions
            # Implementation similar to TLITE
            
            return candidate_logits, offset_logits
    
    # Create model
    twilight_model = Twilight(config)
    
    # Initialize with weights from Voyager
    # This would need to be adapted based on the actual structure differences
    
    # Return the converted model
    return twilight_model


def analyze_clustering_quality(cluster_map, page_metadata):
    '''
    Analyze the quality of clustering
    
    Args:
        cluster_map: Mapping from page IDs to cluster IDs
        page_metadata: Dictionary of page metadata
        
    Returns:
        Dictionary with clustering quality metrics
    '''
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    
    # Extract offset transitions for pages
    page_transitions = []
    page_to_cluster = []
    
    for page_id, metadata in page_metadata.items():
        if page_id in cluster_map:
            # Skip pages with no transitions
            if np.sum(metadata['offset_transitions']) == 0:
                continue
                
            # Normalize transitions
            transitions = metadata['offset_transitions'] / (np.sum(metadata['offset_transitions']) + 1e-10)
            
            # Flatten for distance calculation
            page_transitions.append(transitions.flatten())
            page_to_cluster.append(cluster_map[page_id])
    
    # Convert to numpy arrays
    page_transitions = np.array(page_transitions)
    page_to_cluster = np.array(page_to_cluster)
    
    # Calculate pairwise distances
    distances = squareform(pdist(page_transitions, 'euclidean'))
    
    # Calculate intra-cluster and inter-cluster distances
    intra_cluster_distances = []
    inter_cluster_distances = []
    
    for i in range(len(page_transitions)):
        cluster_i = page_to_cluster[i]
        
        # Intra-cluster distances
        same_cluster = page_to_cluster == cluster_i
        if np.sum(same_cluster) > 1:  # At least one other page in the cluster
            intra_distances = distances[i][same_cluster]
            intra_cluster_distances.append(np.mean(intra_distances[intra_distances > 0]))  # Exclude self-distance
        
        # Inter-cluster distances
        diff_cluster = page_to_cluster != cluster_i
        if np.sum(diff_cluster) > 0:  # At least one page in a different cluster
            inter_distances = distances[i][diff_cluster]
            inter_cluster_distances.append(np.mean(inter_distances))
    
    # Calculate silhouette score
    silhouette_scores = []
    for i in range(len(page_transitions)):
        if i < len(intra_cluster_distances) and i < len(inter_cluster_distances):
            a = intra_cluster_distances[i]
            b = inter_cluster_distances[i]
            silhouette = (b - a) / max(a, b)
            silhouette_scores.append(silhouette)
    
    # Return quality metrics
    return {
        'avg_intra_cluster_distance': np.mean(intra_cluster_distances) if intra_cluster_distances else float('nan'),
        'avg_inter_cluster_distance': np.mean(inter_cluster_distances) if inter_cluster_distances else float('nan'),
        'silhouette_score': np.mean(silhouette_scores) if silhouette_scores else float('nan'),
        'num_clusters_used': len(np.unique(page_to_cluster)),
        'num_pages_clustered': len(page_to_cluster)
    }
