import argparse
import numpy as np
import tensorflow as tf
import lzma
import os
import random

from script.config import TLITEConfig, load_config
from script.model import create_tlite_model
from script.prefetcher import TLITEPrefetcher

def parse_args():
    parser = argparse.ArgumentParser(description='Generate prefetch files using T-LITE model')
    parser.add_argument('--model-path', help='Path to model checkpoint', required=True)
    parser.add_argument('--clustering-path', help='Path to clustering information', required=True)
    parser.add_argument('--benchmark', help='Path to benchmark trace', required=True)
    parser.add_argument('--output', help='Path to output prefetch file', required=True)
    parser.add_argument('--config', default='./config/TLITE2debug1.yaml', help='Path to configuration file')
    parser.add_argument('--prefetch-distance', type=int, default=0, help='Prefetch distance in instructions (0 = no adjustment)')
    parser.add_argument('--debug-file', help='Path to debug log file', default=None)
    parser.add_argument('--simple-mode', action='store_true', help='Use simplified prefetch generation logic')
    return parser.parse_args()

def generate_prefetches_simple(model, clustering_info, trace_path, config, output_path, prefetch_distance=0, debug_file=None):
    """
    Generate prefetch file using T-LITE model with simplified logic
    """
    print(f"\n=== Starting Simple Prefetch Generation ===")
    print(f"Processing trace file: {trace_path}")
    print(f"Output file: {output_path}")
    print(f"Prefetch distance: {prefetch_distance}")
    
    # Create debug file handler if specified
    debug_f = None
    if debug_file:
        os.makedirs(os.path.dirname(debug_file), exist_ok=True)
        debug_f = open(debug_file, 'w')
        print(f"Debug info will be written to: {debug_file}")
    
    # Get page to cluster mapping
    page_to_cluster = clustering_info.get('page_to_cluster', {})
    
    # Create reverse mapping: cluster to pages
    cluster_to_pages = {}
    for page, cluster in page_to_cluster.items():
        if cluster not in cluster_to_pages:
            cluster_to_pages[cluster] = []
        cluster_to_pages[cluster].append(page)
    
    # Initialize statistics
    stats = {
        'total_accesses': 0,
        'model_predictions': 0,
        'no_prefetch_predictions': 0,
        'valid_prefetches': 0,
        'fallback_to_current_page': 0,
        'next_line_prefetches': 0,
        'no_cluster_mapping': 0,
        'empty_candidate_pages': 0,
        'candidate_index_out_of_range': 0
    }
    
    # Initialize history buffers
    history_length = config.history_length
    page_history = [0] * history_length
    offset_history = [0] * history_length
    
    # Create a simple DPF vector (all zeros)
    dpf_vector = np.zeros((1, config.num_candidates), dtype=np.float32)
    
    # Create debug sample set - track these in detail
    debug_samples = set()
    
    # Open files
    if trace_path.endswith('.txt.xz'):
        f = lzma.open(trace_path, mode='rt', encoding='utf-8')
    else:
        f = open(trace_path, 'r')
    
    # Count total lines for progress reporting
    print("Calculating total lines...")
    total_lines = sum(1 for line in f if not (line.startswith('***') or line.startswith('Read')))
    f.seek(0)
    print(f"Total lines to process: {total_lines}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process trace file
    with open(output_path, 'w', buffering=1024*1024) as out_f:
        line_count = 0
        
        for line in f:
            if line.startswith('***') or line.startswith('Read'):
                continue
            
            # Parse line
            split = line.strip().split(', ')
            inst_id = int(split[0])
            pc = int(split[3], 16)
            addr = int(split[2], 16)
            
            # Calculate page and offset
            page = (addr >> 6) >> config.offset_bits
            offset = (addr >> 6) & ((1 << config.offset_bits) - 1)
            
            # Update statistics
            stats['total_accesses'] += 1
            
            # Check if this is a debug sample
            is_debug_sample = (line_count % 10000 == 0) or (len(debug_samples) < 100 and random.random() < 0.01)
            if is_debug_sample:
                debug_samples.add(inst_id)
                
                if debug_f:
                    debug_f.write(f"\n===== Debug Sample {inst_id} =====\n")
                    debug_f.write(f"Line: {line.strip()}\n")
                    debug_f.write(f"PC: {pc:x}, Address: {addr:x}\n")
                    debug_f.write(f"Page: {page:x}, Offset: {offset}\n")
                    debug_f.write(f"Current History - Pages: {page_history}, Offsets: {offset_history}\n")
            
            # Update history
            page_history.pop(0)
            page_history.append(page)
            offset_history.pop(0)
            offset_history.append(offset)
            
            # Prepare model inputs
            # Map pages to clusters
            cluster_history = []
            for p in page_history:
                # Use default of 0 if page not in mapping
                cluster = page_to_cluster.get(p, 0)
                cluster_history.append(cluster)
            
            cluster_input = tf.convert_to_tensor([cluster_history], dtype=tf.int32)
            offset_input = tf.convert_to_tensor([offset_history], dtype=tf.int32)
            pc_input = tf.convert_to_tensor([[pc]], dtype=tf.int32)
            dpf_input = tf.convert_to_tensor(dpf_vector, dtype=tf.float32)
            
            # Make prediction
            candidate_logits, offset_logits = model((cluster_input, offset_input, pc_input, dpf_input))
            
            # Get predicted candidate and offset
            candidate_pred = tf.argmax(candidate_logits[0]).numpy()
            offset_pred = tf.argmax(offset_logits[0]).numpy()
            
            if is_debug_sample and debug_f:
                debug_f.write(f"Predicted - Candidate: {candidate_pred}, Offset: {offset_pred}\n")
                debug_f.write(f"Candidate Logits: {candidate_logits[0].numpy()}\n")
                debug_f.write(f"Offset Logits: {offset_logits[0].numpy()}\n")
            
            # Check if "no prefetch" was predicted
            if candidate_pred == config.num_candidates:
                stats['no_prefetch_predictions'] += 1
                if is_debug_sample and debug_f:
                    debug_f.write("No prefetch predicted\n")
                continue
            
            stats['model_predictions'] += 1
            
            # Get the trigger cluster ID (current page's cluster)
            trigger_cluster = cluster_history[-1]
            if is_debug_sample and debug_f:
                debug_f.write(f"Trigger Cluster: {trigger_cluster}\n")
            
            # Generate prefetch address
            prefetch_addr = None
            
            # Try to map cluster to page
            if trigger_cluster in cluster_to_pages and len(cluster_to_pages[trigger_cluster]) > 0:
                # Choose the most representative page for this cluster
                prefetch_page = cluster_to_pages[trigger_cluster][0]
                prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (offset_pred << 6)
                stats['valid_prefetches'] += 1
                
                if is_debug_sample and debug_f:
                    debug_f.write(f"Successfully mapped cluster {trigger_cluster} to page {prefetch_page:x}\n")
                    debug_f.write(f"Generated prefetch address: {prefetch_addr:x}\n")
            else:
                # No mapping for this cluster, use fallback strategies
                stats['no_cluster_mapping'] += 1
                
                # Fallback strategy 1: Use current page with predicted offset
                prefetch_page = page
                prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (offset_pred << 6)
                stats['fallback_to_current_page'] += 1
                
                if is_debug_sample and debug_f:
                    debug_f.write(f"No mapping for cluster {trigger_cluster}, using current page {prefetch_page:x}\n")
                    debug_f.write(f"Fallback prefetch address: {prefetch_addr:x}\n")
                
                # If the predicted address is the same as current address, use next cache line
                if prefetch_addr == addr:
                    prefetch_addr = addr + 64
                    stats['next_line_prefetches'] += 1
                    
                    if is_debug_sample and debug_f:
                        debug_f.write(f"Prefetch equals current address, using next line: {prefetch_addr:x}\n")
            
            # Apply prefetch distance
            trigger_id = inst_id
            if prefetch_distance > 0:
                trigger_id = max(0, inst_id - prefetch_distance)
                
                if is_debug_sample and debug_f:
                    debug_f.write(f"Applied prefetch distance: {inst_id} -> {trigger_id}\n")
            
            # Write prefetch to file
            out_f.write(f"{trigger_id} {prefetch_addr:x}\n")
            
            if is_debug_sample and debug_f:
                debug_f.write(f"Final output: {trigger_id} {prefetch_addr:x}\n")
                debug_f.write("================================\n")
            
            # Update progress periodically
            line_count += 1
            if line_count % 100000 == 0:
                progress = (line_count / total_lines) * 100
                print(f"Progress: {line_count}/{total_lines} ({progress:.2f}%)")
                
                # Print interim statistics
                prefetch_rate = (stats['valid_prefetches'] + stats['fallback_to_current_page']) / stats['total_accesses'] * 100
                print(f"Current prefetch rate: {prefetch_rate:.2f}%")
    
    # Close debug file if opened
    if debug_f:
        debug_f.close()
    
    # Print final statistics
    print("\n=== Final Statistics ===")
    print(f"Total accesses: {stats['total_accesses']}")
    print(f"Model predictions (non-no-prefetch): {stats['model_predictions']} ({stats['model_predictions']/stats['total_accesses']*100:.2f}%)")
    print(f"No-prefetch predictions: {stats['no_prefetch_predictions']} ({stats['no_prefetch_predictions']/stats['total_accesses']*100:.2f}%)")
    print(f"Valid prefetches: {stats['valid_prefetches']} ({stats['valid_prefetches']/stats['model_predictions']*100:.2f}% of predictions)")
    print(f"Fallback prefetches: {stats['fallback_to_current_page']} ({stats['fallback_to_current_page']/stats['model_predictions']*100:.2f}% of predictions)")
    print(f"Next-line prefetches: {stats['next_line_prefetches']} ({stats['next_line_prefetches']/stats['model_predictions']*100:.2f}% of predictions)")
    print(f"No cluster mapping cases: {stats['no_cluster_mapping']} ({stats['no_cluster_mapping']/stats['model_predictions']*100:.2f}% of predictions)")
    print(f"Total prefetches generated: {stats['valid_prefetches'] + stats['fallback_to_current_page']}")

def generate_prefetches(model, clustering_info, trace_path, config, output_path, prefetch_distance=0, debug_file=None):
    """
    Generate prefetch file using T-LITE model with optimized stream processing
    """
    print(f"\n=== Starting Prefetch Generation ===")
    print(f"Processing trace file: {trace_path}")
    print(f"Output file: {output_path}")
    print(f"Prefetch distance: {prefetch_distance}")
    
    # Create debug file handler if specified
    debug_f = None
    if debug_file:
        os.makedirs(os.path.dirname(debug_file), exist_ok=True)
        debug_f = open(debug_file, 'w')
        print(f"Debug info will be written to: {debug_file}")
    
    # Analyze clustering information
    page_to_cluster = clustering_info.get('page_to_cluster', {})
    if page_to_cluster:
        cluster_pages = list(page_to_cluster.keys())
        min_page = min(cluster_pages)
        max_page = max(cluster_pages)
        print(f"\n=== Clustering Information Analysis ===")
        print(f"Page ID Range: {min_page:x} - {max_page:x}")
        print(f"Number of Clusters: {len(set(page_to_cluster.values()))}")
        print(f"Number of Mapped Pages: {len(page_to_cluster)}")
    
    # Create reverse mapping: cluster to pages
    cluster_to_pages = {}
    for page, cluster in page_to_cluster.items():
        if cluster not in cluster_to_pages:
            cluster_to_pages[cluster] = []
        cluster_to_pages[cluster].append(page)
    
    # Create dynamic cluster mapper
    class DynamicClusterMapper:
        def __init__(self, static_mapping, num_clusters):
            self.static_mapping = static_mapping
            self.dynamic_mapping = {}
            self.num_clusters = num_clusters
            self.stats = {'static_hits': 0, 'dynamic_hits': 0, 'new_mappings': 0}
        
        def get_cluster(self, page_id):
            # Try static mapping
            if page_id in self.static_mapping:
                self.stats['static_hits'] += 1
                return self.static_mapping[page_id]
            
            # Try dynamic mapping
            if page_id in self.dynamic_mapping:
                self.stats['dynamic_hits'] += 1
                return self.dynamic_mapping[page_id]
            
            # Create new mapping
            new_cluster = hash(page_id) % self.num_clusters
            self.dynamic_mapping[page_id] = new_cluster
            self.stats['new_mappings'] += 1
            return new_cluster
        
        def print_stats(self):
            total = sum(self.stats.values())
            if total > 0:
                print("\nMapping Statistics:")
                print(f"- Static Mapping Hits: {self.stats['static_hits']} ({self.stats['static_hits']/total*100:.2f}%)")
                print(f"- Dynamic Mapping Hits: {self.stats['dynamic_hits']} ({self.stats['dynamic_hits']/total*100:.2f}%)")
                print(f"- New Mappings: {self.stats['new_mappings']} ({self.stats['new_mappings']/total*100:.2f}%)")
    
    # Add stream load balancer monitoring
    class StreamLoadBalancer:
        def __init__(self, num_streams):
            self.num_streams = num_streams
            self.stream_loads = np.zeros(num_streams, dtype=np.int64)
            self.total_updates = 0
            self.last_rebalance = 0
            self.rebalance_interval = 100000
        
        def update(self, stream_id):
            self.stream_loads[stream_id] += 1
            self.total_updates += 1
            
            if self.total_updates - self.last_rebalance >= self.rebalance_interval:
                self.check_balance()
                self.last_rebalance = self.total_updates
        
        def check_balance(self):
            if self.total_updates == 0:
                return
            
            active_streams = np.where(self.stream_loads > 0)[0]
            if len(active_streams) == 0:
                return
            
            loads = self.stream_loads[active_streams]
            avg_load = np.mean(loads)
            max_load = np.max(loads)
            min_load = np.min(loads)
            imbalance = max_load / min_load if min_load > 0 else float('inf')
            
            if imbalance > 10:
                print(f"\nWarning: Severe Load Imbalance Detected")
                print(f"- Maximum Load: {max_load}")
                print(f"- Minimum Load: {min_load}")
                print(f"- Average Load: {avg_load:.0f}")
                print(f"- Imbalance Ratio: {imbalance:.2f}x")
                print("Consider adjusting stream allocation algorithm or increasing stream count")
    
    # Reduced stream count for better management
    num_streams = 16
    print(f"Using {num_streams} parallel streams")
    
    # Stream ID calculation function
    def calculate_stream_id(pc, addr):
        """Use improved hash function to calculate stream ID"""
        # Use multiple features for hash
        pc_low = pc & 0xFFFF
        pc_high = (pc >> 16) & 0xFFFF
        addr_low = addr & 0xFFFF
        addr_high = (addr >> 16) & 0xFFFF
        
        # Use FNV-1a hash algorithm
        hash_value = 2166136261
        for value in [pc_low, pc_high, addr_low, addr_high]:
            hash_value = hash_value ^ value
            hash_value = (hash_value * 16777619) & 0xFFFFFFFF
        
        # Use Wang hash for final mixing
        hash_value = hash_value ^ (hash_value >> 16)
        hash_value = (hash_value * 0x85ebca6b) & 0xFFFFFFFF
        hash_value = hash_value ^ (hash_value >> 13)
        hash_value = (hash_value * 0xc2b2ae35) & 0xFFFFFFFF
        hash_value = hash_value ^ (hash_value >> 16)
        
        return hash_value % num_streams
    
    # Reduced batch size for better management
    batch_size = 5000
    print(f"Batch Processing Size: {batch_size}")
    
    # Preallocate batch processing arrays
    max_batch_size = batch_size + 1000
    pages_batch = np.zeros(max_batch_size, dtype=np.int64)
    offsets_batch = np.zeros(max_batch_size, dtype=np.int32)
    pcs_batch = np.zeros(max_batch_size, dtype=np.int64)
    inst_ids_batch = np.zeros(max_batch_size, dtype=np.int64)
    stream_ids_batch = np.zeros(max_batch_size, dtype=np.int32)
    
    # Preallocate model input arrays
    cluster_histories = [[] for _ in range(max_batch_size)]
    offset_histories = [[] for _ in range(max_batch_size)]
    dpf_vectors_batch = [[] for _ in range(max_batch_size)]
    
    # Initialize load balancer
    load_balancer = StreamLoadBalancer(num_streams)
    
    # Initialize prefetchers and mappers
    prefetchers = []
    mappers = []
    for i in range(num_streams):
        prefetcher = TLITEPrefetcher(
            model=model,
            clustering_info=clustering_info,
            config=config
        )
        mapper = DynamicClusterMapper(page_to_cluster, config.num_clusters)
        prefetchers.append(prefetcher)
        mappers.append(mapper)
    
    # TensorFlow batch processing optimization
    @tf.function(experimental_relax_shapes=True, jit_compile=True)
    def predict_batch(cluster_histories, offset_histories, pcs, dpf_vectors):
        return model((cluster_histories, offset_histories, pcs, dpf_vectors))
    
    # Read file and calculate total line count
    if trace_path.endswith('.txt.xz'):
        f = lzma.open(trace_path, mode='rt', encoding='utf-8')
    else:
        f = open(trace_path, 'r')
    
    print("Calculating total lines...")
    total_lines = sum(1 for line in f if not (line.startswith('***') or line.startswith('Read')))
    f.seek(0)
    print(f"Total lines to process: {total_lines}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create statistics counters
    stats = {
        'total_accesses': 0,
        'model_predictions': 0,
        'no_prefetch_predictions': 0,
        'empty_candidate_pages': 0,
        'candidate_index_out_of_range': 0,
        'fallback_to_current_page': 0,
        'valid_prefetches': 0,
        'next_line_prefetches': 0
    }
    
    # Create debug sample set
    debug_samples = set()
    if debug_f:
        # Choose some samples for detailed debugging
        debug_samples = set(random.sample(range(total_lines), min(1000, total_lines)))
    
    # Main processing loop
    current_batch_size = 0
    processed_lines = 0
    total_predictions = 0
    valid_prefetches = 0
    stream_stats = {i: {'updates': 0, 'prefetches': 0} for i in range(num_streams)}
    
    with open(output_path, 'w', buffering=1024*1024) as out_f:
        for line in f:
            if line.startswith('***') or line.startswith('Read'):
                continue
            
            # Parse line
            split = line.strip().split(', ')
            inst_id = int(split[0])
            pc = int(split[3], 16)
            addr = int(split[2], 16)
            
            # Calculate page and offset
            page = (addr >> 6) >> config.offset_bits
            offset = (addr >> 6) & ((1 << config.offset_bits) - 1)
            
            # Check if this is a debug sample
            is_debug_sample = processed_lines in debug_samples
            if is_debug_sample and debug_f:
                debug_f.write(f"\n===== Debug Sample {processed_lines} (Inst ID: {inst_id}) =====\n")
                debug_f.write(f"Line: {line.strip()}\n")
                debug_f.write(f"PC: {pc:x}, Address: {addr:x}\n")
                debug_f.write(f"Page: {page:x}, Offset: {offset}\n")
            
            # Update stats
            stats['total_accesses'] += 1
            
            # Use stream allocation function
            stream_id = calculate_stream_id(pc, addr)
            
            # Update stream statistics
            stream_stats[stream_id]['updates'] += 1
            
            # Update preallocated arrays
            pages_batch[current_batch_size] = page
            offsets_batch[current_batch_size] = offset
            pcs_batch[current_batch_size] = pc
            inst_ids_batch[current_batch_size] = inst_id
            stream_ids_batch[current_batch_size] = stream_id
            
            # Update prefetchers state
            prefetcher = prefetchers[stream_id]
            mapper = mappers[stream_id]
            cluster_id = mapper.get_cluster(page)
            prefetcher.update_history(cluster_id, offset, pc)
            
            # Debug information
            if is_debug_sample and debug_f:
                debug_f.write(f"Mapped to Stream: {stream_id}\n")
                debug_f.write(f"Page {page:x} mapped to Cluster: {cluster_id}\n")
                debug_f.write(f"Updated History - Pages: {prefetcher.page_history}, Offsets: {prefetcher.offset_history}\n")
            
            # Update load balancer
            load_balancer.update(stream_id)
            
            current_batch_size += 1
            
            # Process full batch
            if current_batch_size >= batch_size:
                # Prepare model inputs
                valid_indices = []
                for i in range(current_batch_size):
                    stream_id = stream_ids_batch[i]
                    prefetcher = prefetchers[stream_id]
                    
                    cluster_history, offset_history, _, dpf_vectors = prefetcher.metadata_manager.prepare_model_inputs(
                        prefetcher.page_history,
                        prefetcher.offset_history,
                        pcs_batch[i],
                        prefetcher.cluster_mapping_utils
                    )
                    
                    cluster_histories[i] = cluster_history
                    offset_histories[i] = offset_history
                    dpf_vectors_batch[i] = dpf_vectors
                    valid_indices.append(i)
                    
                    # Debug information
                    is_debug_sample = processed_lines + i in debug_samples
                    if is_debug_sample and debug_f:
                        debug_f.write(f"Model Inputs for Sample {processed_lines + i}:\n")
                        debug_f.write(f"  Cluster History: {cluster_history}\n")
                        debug_f.write(f"  Offset History: {offset_history}\n")
                        debug_f.write(f"  PC: {pcs_batch[i]:x}\n")
                        debug_f.write(f"  DPF Vectors: {dpf_vectors}\n")
                
                # Batch prediction
                if valid_indices:
                    candidate_logits, offset_logits = predict_batch(
                        tf.convert_to_tensor(cluster_histories[:current_batch_size], dtype=tf.int32),
                        tf.convert_to_tensor(offset_histories[:current_batch_size], dtype=tf.int32),
                        tf.convert_to_tensor(pcs_batch[:current_batch_size], dtype=tf.int32),
                        tf.convert_to_tensor(dpf_vectors_batch[:current_batch_size], dtype=tf.float32)
                    )
                    
                    # Process prediction results
                    candidates = tf.argmax(candidate_logits, axis=1).numpy()
                    offsets = tf.argmax(offset_logits, axis=1).numpy()
                    
                    # Batch generate prefetches
                    for j, i in enumerate(valid_indices):
                        is_debug_sample = processed_lines + i in debug_samples
                        
                        if is_debug_sample and debug_f:
                            debug_f.write(f"Prediction Results for Sample {processed_lines + i}:\n")
                            debug_f.write(f"  Candidate Prediction: {candidates[j]}\n")
                            debug_f.write(f"  Offset Prediction: {offsets[j]}\n")
                            debug_f.write(f"  Candidate Logits: {candidate_logits[j].numpy()}\n")
                            debug_f.write(f"  Offset Logits: {offset_logits[j].numpy()}\n")
                        
                        if candidates[j] == config.num_candidates:
                            stats['no_prefetch_predictions'] += 1
                            if is_debug_sample and debug_f:
                                debug_f.write("  No prefetch predicted\n")
                            continue
                        
                        stats['model_predictions'] += 1
                        
                        stream_id = stream_ids_batch[i]
                        prefetcher = prefetchers[stream_id]
                        candidate_pages = prefetcher.metadata_manager.get_candidate_pages(
                            prefetcher.page_history[-1]
                        )
                        
                        if is_debug_sample and debug_f:
                            debug_f.write(f"  Candidate Pages: {candidate_pages}\n")
                        
                        # Determine prefetch page
                        prefetch_page = None
                        if not candidate_pages:
                            stats['empty_candidate_pages'] += 1
                            prefetch_page = pages_batch[i]
                            stats['fallback_to_current_page'] += 1
                            
                            if is_debug_sample and debug_f:
                                debug_f.write("  Empty candidate pages, falling back to current page\n")
                        elif candidates[j] >= len(candidate_pages):
                            stats['candidate_index_out_of_range'] += 1
                            prefetch_page = pages_batch[i]
                            stats['fallback_to_current_page'] += 1
                            
                            if is_debug_sample and debug_f:
                                debug_f.write("  Candidate index out of range, falling back to current page\n")
                        else:
                            prefetch_page = candidate_pages[candidates[j]][0]
                            stats['valid_prefetches'] += 1
                            
                            if is_debug_sample and debug_f:
                                debug_f.write(f"  Selected prefetch page: {prefetch_page:x}\n")
                        
                        # Calculate prefetch address
                        prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (offsets[j] << 6)
                        curr_addr = (pages_batch[i] << (config.offset_bits + 6)) | (offsets_batch[i] << 6)
                        
                        # Check if prefetch is same as current address
                        if prefetch_addr == curr_addr:
                            prefetch_addr = curr_addr + 64  # Next cache line
                            stats['next_line_prefetches'] += 1
                            
                            if is_debug_sample and debug_f:
                                debug_f.write("  Prefetch equals current address, using next line\n")
                        
                        # Apply prefetch distance if specified
                        trigger_id = inst_ids_batch[i]
                        if prefetch_distance > 0:
                            trigger_id = max(0, inst_ids_batch[i] - prefetch_distance)
                            
                            if is_debug_sample and debug_f:
                                debug_f.write(f"  Applied prefetch distance: {inst_ids_batch[i]} -> {trigger_id}\n")
                        
                        # Write prefetch to file
                        out_f.write(f"{trigger_id} {prefetch_addr:x}\n")
                        
                        if is_debug_sample and debug_f:
                            debug_f.write(f"  Final output: {trigger_id} {prefetch_addr:x}\n")
                        
                        stream_stats[stream_id]['prefetches'] += 1
                        valid_prefetches += 1
                    
                    total_predictions += len(valid_indices)
                
                # Reset batch processing counter
                current_batch_size = 0
                processed_lines += batch_size
                
                # Print progress and statistics
                if processed_lines % (batch_size * 10) == 0:
                    progress = (processed_lines / total_lines) * 100
                    print(f"Progress: {processed_lines}/{total_lines} ({progress:.2f}%)")
                    
                    # Print interim statistics
                    if total_predictions > 0:
                        prefetch_rate = valid_prefetches / total_predictions * 100
                        print(f"Current Prefetch Rate: {prefetch_rate:.2f}%")
                        
                        # Print detailed statistics
                        print("Current Statistics:")
                        print(f"- No Prefetch Predictions: {stats['no_prefetch_predictions']}")
                        print(f"- Empty Candidate Pages: {stats['empty_candidate_pages']}")
                        print(f"- Candidate Index Out of Range: {stats['candidate_index_out_of_range']}")
                        print(f"- Fallback to Current Page: {stats['fallback_to_current_page']}")
                        print(f"- Valid Prefetches: {stats['valid_prefetches']}")
                        print(f"- Next Line Prefetches: {stats['next_line_prefetches']}")
    
    # Close debug file if opened
    if debug_f:
        debug_f.close()
    
    # Print mapper statistics
    for i, mapper in enumerate(mappers):
        print(f"\nMapper {i} Statistics:")
        mapper.print_stats()
    
    # Final Statistics
    print("\n=== Final Statistics ===")
    print(f"Total Lines Processed: {processed_lines}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Valid Prefetches: {valid_prefetches}")
    if total_predictions > 0:
        print(f"Overall Prefetch Rate: {valid_prefetches/total_predictions*100:.2f}%")
    else:
        print("Overall Prefetch Rate: 0.00% (no predictions)")
    
    # Detailed statistics
    print("\n=== Detailed Statistics ===")
    print(f"Total Accesses: {stats['total_accesses']}")
    print(f"Model Predictions: {stats['model_predictions']} ({stats['model_predictions']/stats['total_accesses']*100:.2f}%)")
    print(f"No Prefetch Predictions: {stats['no_prefetch_predictions']} ({stats['no_prefetch_predictions']/stats['total_accesses']*100:.2f}%)")
    print(f"Empty Candidate Pages: {stats['empty_candidate_pages']} ({stats['empty_candidate_pages']/stats['model_predictions']*100 if stats['model_predictions'] > 0 else 0:.2f}%)")
    print(f"Candidate Index Out of Range: {stats['candidate_index_out_of_range']} ({stats['candidate_index_out_of_range']/stats['model_predictions']*100 if stats['model_predictions'] > 0 else 0:.2f}%)")
    print(f"Fallback to Current Page: {stats['fallback_to_current_page']} ({stats['fallback_to_current_page']/stats['model_predictions']*100 if stats['model_predictions'] > 0 else 0:.2f}%)")
    print(f"Valid Prefetches: {stats['valid_prefetches']} ({stats['valid_prefetches']/stats['model_predictions']*100 if stats['model_predictions'] > 0 else 0:.2f}%)")
    print(f"Next Line Prefetches: {stats['next_line_prefetches']} ({stats['next_line_prefetches']/stats['model_predictions']*100 if stats['model_predictions'] > 0 else 0:.2f}%)")
    
    # Print final stream load distribution
    print("\n=== Final Stream Load Distribution ===")
    total_updates = sum(stats['updates'] for stats in stream_stats.values())
    if total_updates > 0:
        active_streams = [(sid, stats) for sid, stats in stream_stats.items() 
                       if stats['updates'] > 0]
        sorted_streams = sorted(active_streams, 
                             key=lambda x: x[1]['updates'], 
                             reverse=True)
        
        print("\nTop 10 Most Active Streams:")
        for stream_id, stats in sorted_streams[:10]:
            prefetch_rate = (stats['prefetches'] / stats['updates'] * 100 
                           if stats['updates'] > 0 else 0)
            print(f"Stream {stream_id}:")
            print(f"   Load: {stats['updates']/total_updates*100:.2f}% ({stats['updates']} updates)")
            print(f"   Prefetches: {prefetch_rate:.2f}% ({stats['prefetches']} prefetches)")
        
        print("\nLoad Distribution Statistics:")
        active_count = len(active_streams)
        print(f"Active Streams: {active_count}/{num_streams}")
        if active_count > 0:
            avg_load = total_updates / active_count
            max_load = max(stats['updates'] for _, stats in active_streams)
            min_load = min(stats['updates'] for _, stats in active_streams)
            print(f"Average Load: {avg_load:.0f} updates/stream")
            print(f"Load Range: {min_load} - {max_load} updates")
            print(f"Max/Min Load Ratio: {max_load/min_load:.2f}x")

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model, ignore warnings and add debug output
    print("\n=== Loading Model ===")
    model = create_tlite_model(config)
    model.load_weights(args.model_path).expect_partial()
    print(f"Model weights loaded successfully. Model structure:")
    print(model.summary())
    
    # Load clustering information with debug output
    print("\n=== Loading Clustering Information ===")
    clustering_info = np.load(args.clustering_path, allow_pickle=True).item()
    print(f"Clustering information loaded. Available keys: {clustering_info.keys()}")
    print(f"Clustering information contains {len(clustering_info.get('page_to_cluster', {}))} page mappings")

    # Modify random sampling print code
    page_to_cluster = clustering_info.get('page_to_cluster', {})
    if page_to_cluster:
        sample_keys = random.sample(list(page_to_cluster.keys()), min(5, len(page_to_cluster)))
        print("Random sample of 5 mappings:")
        for page in sample_keys:
            cluster = page_to_cluster[page]
            if isinstance(page, int):
                print(f"Page {page:x} -> Cluster {cluster}")
            else:
                print(f"Page {page} -> Cluster {cluster}")
    
    # Generate prefetches using selected mode
    if args.simple_mode:
        generate_prefetches_simple(
            model=model,
            clustering_info=clustering_info,
            trace_path=args.benchmark,
            config=config,
            output_path=args.output,
            prefetch_distance=args.prefetch_distance,
            debug_file=args.debug_file
        )
    else:
        generate_prefetches(
            model=model,
            clustering_info=clustering_info,
            trace_path=args.benchmark,
            config=config,
            output_path=args.output,
            prefetch_distance=args.prefetch_distance,
            debug_file=args.debug_file
        )
    
    # Generate multiple prefetch files with different distances if prefetch_distance is 0
    if args.prefetch_distance == 0:
        print("\n=== Generating Multiple Prefetch Files with Different Distances ===")
        distances = [20, 50, 100, 200, 500]
        
        for distance in distances:
            distance_output = f"{os.path.splitext(args.output)[0]}_d{distance}{os.path.splitext(args.output)[1]}"
            distance_debug = f"{os.path.splitext(args.debug_file)[0]}_d{distance}.log" if args.debug_file else None
            
            print(f"\nGenerating prefetch file with distance {distance}...")
            if args.simple_mode:
                generate_prefetches_simple(
                    model=model,
                    clustering_info=clustering_info,
                    trace_path=args.benchmark,
                    config=config,
                    output_path=distance_output,
                    prefetch_distance=distance,
                    debug_file=distance_debug
                )
            else:
                generate_prefetches(
                    model=model,
                    clustering_info=clustering_info,
                    trace_path=args.benchmark,
                    config=config,
                    output_path=distance_output,
                    prefetch_distance=distance,
                    debug_file=distance_debug
                )
    
    print("Prefetch generation complete!")

if __name__ == "__main__":
    main()