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
    return parser.parse_args()

def generate_prefetches(model, clustering_info, trace_path, config, output_path):
    """
    Generate prefetch file using T-LITE model with optimized stream processing
    """
    print(f"\n=== Starting Prefetch Generation ===")
    print(f"Processing trace file: {trace_path}")
    print(f"Output file: {output_path}")
    
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
    
    # Optimization 1: Increase stream count and improve stream allocation
    num_streams = 32
    print(f"Using {num_streams} parallel streams")
    
    # Optimization 2: Improve stream ID calculation function
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
    
    # Optimization 3: Increase batch processing size and use memory preallocation
    batch_size = 50000  # Increase to 50000
    print(f"Batch Processing Size: {batch_size}")
    
    # Optimization 4: Preallocate batch processing arrays to reduce memory allocation
    max_batch_size = batch_size + 1000  # Add some buffer
    pages_batch = np.zeros(max_batch_size, dtype=np.int64)
    offsets_batch = np.zeros(max_batch_size, dtype=np.int32)
    pcs_batch = np.zeros(max_batch_size, dtype=np.int64)
    inst_ids_batch = np.zeros(max_batch_size, dtype=np.int64)
    stream_ids_batch = np.zeros(max_batch_size, dtype=np.int32)
    
    # Optimization 5: Preallocate model input arrays
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
    
    # Optimization 6: Use TensorFlow batch processing optimization
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
            
            # Use new stream allocation function
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
            
            # In main loop update load balancer
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
                
                # Batch prediction
                if valid_indices:
                    candidate_logits, offset_logits = predict_batch(
                        tf.convert_to_tensor(cluster_histories[:current_batch_size], dtype=tf.int32),
                        tf.convert_to_tensor(offset_histories[:current_batch_size], dtype=tf.int32),
                        tf.convert_to_tensor(pcs_batch[:current_batch_size], dtype=tf.int32),
                        tf.convert_to_tensor(dpf_vectors_batch[:current_batch_size], dtype=tf.float32)
                    )
                    
                    # Process prediction results (use vectorized operations)
                    candidates = tf.argmax(candidate_logits, axis=1).numpy()
                    offsets = tf.argmax(offset_logits, axis=1).numpy()
                    
                    # Batch generate prefetches
                    for j, i in enumerate(valid_indices):
                        if candidates[j] != config.num_candidates:
                            stream_id = stream_ids_batch[i]
                            prefetcher = prefetchers[stream_id]
                            candidate_pages = prefetcher.metadata_manager.get_candidate_pages(
                                prefetcher.page_history[-1]
                            )
                            
                            if candidate_pages and candidates[j] < len(candidate_pages):
                                prefetch_page = candidate_pages[candidates[j]][0]
                            else:
                                prefetch_page = pages_batch[i]
                            
                            prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (offsets[j] << 6)
                            out_f.write(f"{inst_ids_batch[i]} {prefetch_addr:x}\n")
                            stream_stats[stream_id]['prefetches'] += 1
                            valid_prefetches += 1
                    
                    total_predictions += len(valid_indices)
                
                # Reset batch processing counter
                current_batch_size = 0
                processed_lines += batch_size
                
                # Print progress and stream load distribution
                if processed_lines % (batch_size * 10) == 0:
                    print(f"Progress: {processed_lines}/{total_lines} ({processed_lines/total_lines*100:.2f}%)")
                    print(f"Current Prefetch Rate: {(valid_prefetches/total_predictions*100) if total_predictions > 0 else 0.00:.2f}%")
                    
                    if processed_lines % (batch_size * 100) == 0:
                        total_updates = sum(stats['updates'] for stats in stream_stats.values())
                        if total_updates > 0:
                            print("\n=== Stream Load Distribution ===")
                            # Only show active streams
                            active_streams = [(sid, stats) for sid, stats in stream_stats.items() 
                                           if stats['updates'] > 0]
                            # Sort by update count
                            sorted_streams = sorted(active_streams, 
                                                 key=lambda x: x[1]['updates'], 
                                                 reverse=True)
                            # Show top 5 most active streams
                            for stream_id, stats in sorted_streams[:5]:
                                print(f"Stream {stream_id}: {stats['updates']/total_updates*100:.2f}% "
                                     f"({stats['updates']} updates)")
                            # Show load distribution statistics
                            active_count = len(active_streams)
                            print(f"\nActive Streams: {active_count}/{num_streams}")
                            if active_count > 0:
                                avg_load = total_updates / active_count
                                max_load = max(stats['updates'] for _, stats in active_streams)
                                min_load = min(stats['updates'] for _, stats in active_streams)
                                print(f"Average Load: {avg_load:.0f} updates/stream")
                                print(f"Load Range: {min_load} - {max_load} updates")
                                print(f"Max/Min Load Ratio: {max_load/min_load:.2f}x")
    
    # Final Statistics
    print("\n=== Final Statistics ===")
    print(f"Total Lines Processed: {processed_lines}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Valid Prefetches: {valid_prefetches}")
    if total_predictions > 0:
        print(f"Overall Prefetch Rate: {valid_prefetches/total_predictions*100:.2f}%")
    else:
        print("Overall Prefetch Rate: 0.00% (no predictions)")
    
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
    
    # Generate prefetches
    generate_prefetches(
        model=model,
        clustering_info=clustering_info,
        trace_path=args.benchmark,
        config=config,
        output_path=args.output
    )
    
    print("Prefetch generation complete!")

if __name__ == "__main__":
    main()