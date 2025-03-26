import argparse
import numpy as np
import tensorflow as tf
import lzma
import os

from script.config import TLITEConfig, load_config, extend_voyager_config_for_tlite
from script.model import create_tlite_model
from script.clustering import BehavioralClusteringUtils

def parse_args():
    parser = argparse.ArgumentParser(description='T-LITE model training')
    parser.add_argument('--benchmark', help='Path to the benchmark trace', required=True)
    parser.add_argument('--model-path', help='Path to save model checkpoint', required=True)
    parser.add_argument('--config', default='./script/config/TLITE2debug1.yaml', help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode with smaller dataset')
    parser.add_argument('--tb-dir', help='TensorBoard log directory')
    return parser.parse_args()

def process_trace_file(trace_path, config, max_lines=None):
    """
    Process trace file into training data for direct T-LITE training
    with behavioral clustering
    """
    print(f"Processing trace file: {trace_path}")
    
    # Data structures for processing
    cluster_sequences = []
    offset_sequences = []
    pc_sequences = []
    next_clusters = []
    next_offsets = []
    candidate_lists = []
    
    # Page to cluster mapping
    page_to_cluster = {}
    current_clusters = 0
    
    # Offset transitions for dynamic clustering
    page_offset_transitions = {}  # {page_id: 64x64 matrix}
    cluster_offset_transitions = [np.zeros((64, 64), dtype=np.float32) 
                                 for _ in range(config.num_clusters)]
    
    # Metadata for frequency-based candidate selection
    cluster_successors = {}  # {cluster_id: {successor_cluster: count}}
    
    # Read trace file
    if trace_path.endswith('.txt.xz'):
        f = lzma.open(trace_path, mode='rt', encoding='utf-8')
    else:
        f = open(trace_path, 'r')
        
    # Process line by line
    clusters = []
    offsets = []
    pcs = []
    pages = []  # Keep track of pages for updating transitions
    lines_processed = 0
    
    for line in f:
        # Skip comments
        if line.startswith('***') or line.startswith('Read'):
            continue
            
        # Parse line (format depends on your trace format)
        # For example with MLPrefetchingCompetition format:
        split = line.strip().split(', ')
        inst_id = int(split[0])
        pc = int(split[3], 16)
        addr = int(split[2], 16)
        
        # Convert address to page and offset
        page = (addr >> 6) >> config.offset_bits  # Get page ID
        offset = (addr >> 6) & ((1 << config.offset_bits) - 1)  # Get offset
        
        # Track offset transitions for this page
        if page not in page_offset_transitions:
            page_offset_transitions[page] = np.zeros((64, 64), dtype=np.int32)
        
        # Update offset transitions if we have a previous access
        if len(pages) > 0:
            prev_page = pages[-1]
            prev_offset = offsets[-1]
            page_offset_transitions[prev_page][prev_offset, offset] += 1
        
        # Assign cluster to page if not already assigned
        if page not in page_to_cluster:
            # If we have enough transition data, assign cluster based on behavior
            if page in page_offset_transitions and np.sum(page_offset_transitions[page]) > 5:
                # Find closest cluster based on offset transition patterns
                best_cluster = 0
                min_distance = float('inf')
                
                for cluster_id, cluster_transitions in enumerate(cluster_offset_transitions):
                    # Skip empty clusters
                    if np.sum(cluster_transitions) == 0:
                        continue
                    
                    # Normalize transitions
                    norm_page = page_offset_transitions[page] / (np.sum(page_offset_transitions[page]) + 1e-10)
                    norm_cluster = cluster_transitions / (np.sum(cluster_transitions) + 1e-10)
                    
                    # Calculate distance
                    dist = np.linalg.norm(norm_page - norm_cluster)
                    if dist < min_distance:
                        min_distance = dist
                        best_cluster = cluster_id
                
                # Assign to best matching cluster
                page_to_cluster[page] = best_cluster
            else:
                # Simple method: sequentially assign new clusters until max is reached, then reuse
                if current_clusters < config.num_clusters:
                    cluster_id = current_clusters
                    current_clusters += 1
                else:
                    # Choose a random cluster if we've used all available ones
                    cluster_id = np.random.randint(0, config.num_clusters)
                
                page_to_cluster[page] = cluster_id
        
        # Get cluster ID for this page
        cluster = page_to_cluster[page]
        
        # Update cluster's aggregate offset transitions
        if len(pages) > 0:
            prev_page = pages[-1]
            prev_offset = offsets[-1]
            prev_cluster = page_to_cluster[prev_page]
            cluster_offset_transitions[prev_cluster][prev_offset, offset] += 1
        
        # Store in history
        clusters.append(cluster)
        offsets.append(offset)
        pcs.append(pc)
        pages.append(page)
        
        # Update cluster successor frequencies
        if len(clusters) > 1:
            prev_cluster = clusters[-2]
            if prev_cluster not in cluster_successors:
                cluster_successors[prev_cluster] = {}
            
            successor = cluster
            if successor not in cluster_successors[prev_cluster]:
                cluster_successors[prev_cluster][successor] = 0
            cluster_successors[prev_cluster][successor] += 1
        
        # Create training samples when we have enough history
        if len(clusters) >= config.history_length + 1:
            # Input: last N clusters/offsets/PCs
            cluster_seq = clusters[-config.history_length-1:-1]
            offset_seq = offsets[-config.history_length-1:-1]
            pc_seq = pcs[-config.history_length-1:-1]
            
            # Output: next cluster and offset
            next_cluster = clusters[-1]
            next_offset = offsets[-1]
            
            # Get candidate list for the trigger cluster
            trigger_cluster = cluster_seq[-1]
            if trigger_cluster in cluster_successors:
                candidates = sorted(
                    cluster_successors[trigger_cluster].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:config.num_candidates]
            else:
                candidates = []
            
            # Add to dataset
            cluster_sequences.append(cluster_seq)
            offset_sequences.append(offset_seq)
            pc_sequences.append(pc_seq)
            next_clusters.append(next_cluster)
            next_offsets.append(next_offset)
            candidate_lists.append(candidates)
        
        lines_processed += 1
        if max_lines and lines_processed >= max_lines:
            break
        
        # Print progress periodically
        if lines_processed % 1000000 == 0:
            print(f"Processed {lines_processed} lines...")
    
    f.close()
    
    # Create TensorFlow datasets
    def generator():
        for i in range(len(cluster_sequences)):
            # Find candidate index
            candidate_idx = config.num_candidates  # Default to no-prefetch
            for j, (cand, _) in enumerate(candidate_lists[i]):
                if cand == next_clusters[i]:
                    candidate_idx = j
                    break
            
            # Create DPF vector
            dpf_vector = np.zeros(config.num_candidates, dtype=np.float32)
            if candidate_lists[i]:
                total_freq = sum(freq for _, freq in candidate_lists[i])
                for j, (_, freq) in enumerate(candidate_lists[i]):
                    if j < config.num_candidates:
                        dpf_vector[j] = freq / total_freq
            
            yield (
                np.array(cluster_sequences[i], dtype=np.int32),
                np.array(offset_sequences[i], dtype=np.int32),
                np.array([pc_sequences[i][-1]], dtype=np.int32),
                np.array([dpf_vector], dtype=np.float32),
            ), (
                candidate_idx,
                next_offsets[i]
            )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (
                tf.TensorSpec(shape=(config.history_length,), dtype=tf.int32),
                tf.TensorSpec(shape=(config.history_length,), dtype=tf.int32),
                tf.TensorSpec(shape=(1,), dtype=tf.int32),
                tf.TensorSpec(shape=(1, config.num_candidates), dtype=tf.float32)
            ),
            (
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
    )
    
    # Split into train/validation
    total_samples = len(cluster_sequences)
    train_size = int(0.8 * total_samples)
    train_ds = dataset.take(train_size).shuffle(10000).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = dataset.skip(train_size).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f"Processed {lines_processed} lines, created {total_samples} samples")
    print(f"Train: {train_size}, Validation: {total_samples - train_size}")
    print(f"Used {current_clusters} clusters out of {config.num_clusters} available")
    
    # Save clustering information
    clustering_info = {
        'page_to_cluster': page_to_cluster,
        'cluster_offset_transitions': cluster_offset_transitions,
        'cluster_successors': cluster_successors
    }
    
    return train_ds, valid_ds, clustering_info

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set debug mode if requested
    if args.debug:
        config.epochs = 5
        max_lines = 500000
    else:
        max_lines = None
    
    # Process trace file
    train_ds, valid_ds, clustering_info = process_trace_file(
        args.benchmark, config, max_lines
    )
    
    # Create model
    model = create_tlite_model(config)
    
    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_overall_accuracy',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_overall_accuracy',
            factor=config.lr_decay_rate,
            patience=config.early_stopping_patience // 2,
            min_lr=0.0001,
            mode='max'
        )
    ]
    
    # Add TensorBoard callback if requested
    if args.tb_dir:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=args.tb_dir,
                histogram_freq=1
            )
        )
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model and clustering information
    print(f"Saving model to {args.model_path}")
    model.save_weights(args.model_path)
    
    # Save clustering information
    clustering_path = os.path.join(os.path.dirname(args.model_path), 'clustering.npy')
    np.save(clustering_path, clustering_info, allow_pickle=True)
    
    print("Training complete!")

if __name__ == "__main__":
    main()