import argparse
import numpy as np
import tensorflow as tf
import lzma
import os
import random
import time
from collections import defaultdict

from script.config import TLITEConfig, load_config
from script.model import create_tlite_model
from script.prefetcher import TLITEPrefetcher

def parse_args():
    parser = argparse.ArgumentParser(description='Generate prefetch files using T-LITE model')
    parser.add_argument('--model-path', help='Path to model checkpoint', required=True)
    parser.add_argument('--clustering-path', help='Path to clustering information', required=True)
    parser.add_argument('--benchmark', help='Path to benchmark trace', required=True)
    parser.add_argument('--output', help='Path to output prefetch file', required=True)
    parser.add_argument('--config', default='./config/base1.yaml', help='Path to configuration file')
    parser.add_argument('--prefetch-distance', type=int, default=0, help='Prefetch distance in instructions (0 = no adjustment)')
    parser.add_argument('--debug-file', help='Path to debug log file', default=None)
    parser.add_argument('--simple-mode', action='store_true', help='Use simplified prefetch generation logic')
    parser.add_argument('--sequential', action='store_true', help='Process trace sequentially without batching')
    parser.add_argument('--test-clustering', action='store_true', help='Run tests on clustering mapping')
    return parser.parse_args()

def test_clustering_mapping(clustering_info, config):
    """
    Run diagnostic tests on the clustering mapping to verify it's working correctly
    """
    print("\n=== Testing Clustering Mapping ===")
    page_to_cluster = clustering_info.get('page_to_cluster', {})
    
    if not page_to_cluster:
        print("ERROR: page_to_cluster mapping is empty!")
        return
    
    # Test 1: Check distribution of pages across clusters
    cluster_counts = defaultdict(int)
    for page, cluster in page_to_cluster.items():
        cluster_counts[cluster] += 1
    
    print(f"Total unique pages: {len(page_to_cluster)}")
    print(f"Total unique clusters: {len(cluster_counts)}")
    
    # Find most and least populated clusters
    if cluster_counts:
        max_cluster = max(cluster_counts.items(), key=lambda x: x[1])
        min_cluster = min(cluster_counts.items(), key=lambda x: x[1])
        avg_pages = len(page_to_cluster) / len(cluster_counts)
        
        print(f"Most populated cluster: Cluster {max_cluster[0]} with {max_cluster[1]} pages")
        print(f"Least populated cluster: Cluster {min_cluster[0]} with {min_cluster[1]} pages")
        print(f"Average pages per cluster: {avg_pages:.2f}")
    
    # Test 2: Check cluster ID range
    min_cluster = min(page_to_cluster.values()) if page_to_cluster else None
    max_cluster = max(page_to_cluster.values()) if page_to_cluster else None
    
    print(f"Cluster ID range: {min_cluster} to {max_cluster}")
    print(f"Configured num_clusters: {config.num_clusters}")
    
    if max_cluster is not None and max_cluster >= config.num_clusters:
        print(f"WARNING: Maximum cluster ID ({max_cluster}) exceeds configured num_clusters ({config.num_clusters})")
    
    # Test 3: Check for problematic clusters (if any)
    empty_clusters = [i for i in range(config.num_clusters) if i not in cluster_counts]
    if len(empty_clusters) > 0:
        print(f"WARNING: {len(empty_clusters)} clusters have no pages mapped to them")
        if len(empty_clusters) < 10:
            print(f"Empty clusters: {empty_clusters}")
    
    # Test 4: Create a reverse mapping (cluster to pages) and check its structure
    cluster_to_pages = {}
    for page, cluster in page_to_cluster.items():
        if cluster not in cluster_to_pages:
            cluster_to_pages[cluster] = []
        cluster_to_pages[cluster].append(page)
    
    # Sample a few clusters and their pages
    sampled_clusters = random.sample(list(cluster_to_pages.keys()), min(5, len(cluster_to_pages)))
    print("\nSample cluster to pages mapping:")
    for cluster in sampled_clusters:
        pages = cluster_to_pages[cluster]
        print(f"Cluster {cluster}: {len(pages)} pages, first few: {[f'0x{p:x}' if isinstance(p, int) else p for p in pages[:3]]}...")
    
    return cluster_to_pages  # Return this for potential use in generation

def generate_prefetches_efficient(model, clustering_info, trace_path, config, output_path, prefetch_distance=0, debug_file=None):
    """
    Generate prefetch file using a hybrid approach:
    - Process trace in mini-batches to maintain locality
    - Compile model with tf.function for better performance
    - Use vectorized operations where possible
    """
    print(f"\n=== Starting Efficient Prefetch Generation ===")
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
    
    # Create single prefetcher
    prefetcher = TLITEPrefetcher(
        model=model,
        clustering_info=clustering_info,
        config=config
    )
    
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
    
    mapper = DynamicClusterMapper(page_to_cluster, config.num_clusters)
    
    # Optimize model prediction with tf.function
    @tf.function(jit_compile=True)
    def predict_batch(cluster_histories, offset_histories, pcs, dpf_vectors):
        return model((cluster_histories, offset_histories, pcs, dpf_vectors))
    
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
    
    # Statistics counters
    stats = {
        'total_accesses': 0,
        'model_predictions': 0,
        'no_prefetch_predictions': 0,
        'empty_candidate_pages': 0,
        'candidate_index_out_of_range': 0,
        'fallback_to_current_page': 0,
        'valid_prefetches': 0,
        'next_line_prefetches': 0,
        'processing_time': 0,
        'prediction_time': 0
    }
    
    # Set mini-batch size (small enough to maintain locality, large enough for efficiency)
    mini_batch_size = 128
    
    # Pre-allocate arrays for mini-batch processing
    pages_batch = np.zeros(mini_batch_size, dtype=np.int64)
    offsets_batch = np.zeros(mini_batch_size, dtype=np.int32)
    pcs_batch = np.zeros(mini_batch_size, dtype=np.int64)
    inst_ids_batch = np.zeros(mini_batch_size, dtype=np.int64)
    addrs_batch = np.zeros(mini_batch_size, dtype=np.int64)
    cluster_histories_batch = np.zeros((mini_batch_size, config.history_length), dtype=np.int32)
    offset_histories_batch = np.zeros((mini_batch_size, config.history_length), dtype=np.int32)
    dpf_vectors_batch = np.zeros((mini_batch_size, 1, config.num_candidates), dtype=np.float32)
    
    # Start processing
    start_time = time.time()
    line_count = 0
    debug_sample_indices = set()
    current_batch_size = 0
    
    with open(output_path, 'w', buffering=1024*1024) as out_f:
        # Main processing loop
        while True:
            mini_batch_lines = []
            batch_count = 0
            
            # Read a mini-batch of lines
            while batch_count < mini_batch_size:
                line = f.readline()
                if not line:  # End of file
                    break
                
                if line.startswith('***') or line.startswith('Read'):
                    continue
                
                mini_batch_lines.append(line)
                batch_count += 1
            
            if not mini_batch_lines:  # End of file
                break
            
            current_batch_size = len(mini_batch_lines)
            
            # Process each line in the mini-batch sequentially to maintain history consistency
            for i, line in enumerate(mini_batch_lines):
                # Parse line
                split = line.strip().split(', ')
                inst_id = int(split[0])
                pc = int(split[3], 16)
                addr = int(split[2], 16)
                
                # Calculate page and offset
                page = (addr >> 6) >> config.offset_bits
                offset = (addr >> 6) & ((1 << config.offset_bits) - 1)
                
                # Store in batch arrays
                pages_batch[i] = page
                offsets_batch[i] = offset
                pcs_batch[i] = pc
                inst_ids_batch[i] = inst_id
                addrs_batch[i] = addr
                
                # Select debug samples
                is_debug_sample = (line_count % 10000 == 0) or (random.random() < 0.001)
                if is_debug_sample:
                    debug_sample_indices.add(i)
                
                # Get cluster ID for current page
                cluster_id = mapper.get_cluster(page)
                
                # Update prefetcher history
                prefetcher.update_history(cluster_id, offset, pc)
                
                # Prepare model inputs for this sample
                cluster_history, offset_history, _, dpf_vectors = prefetcher.metadata_manager.prepare_model_inputs(
                    prefetcher.page_history, 
                    prefetcher.offset_history,
                    pc,
                    prefetcher.cluster_mapping_utils
                )
                
                # Store in batch arrays
                cluster_histories_batch[i] = cluster_history
                offset_histories_batch[i] = offset_history
                dpf_vectors_batch[i, 0] = dpf_vectors
                
                line_count += 1
            
            # Batch prediction for all samples
            prediction_start = time.time()
            candidate_logits, offset_logits = predict_batch(
                tf.convert_to_tensor(cluster_histories_batch[:current_batch_size], dtype=tf.int32),
                tf.convert_to_tensor(offset_histories_batch[:current_batch_size], dtype=tf.int32),
                tf.convert_to_tensor(pcs_batch[:current_batch_size, np.newaxis], dtype=tf.int32),
                tf.convert_to_tensor(dpf_vectors_batch[:current_batch_size], dtype=tf.float32)
            )
            stats['prediction_time'] += time.time() - prediction_start
            
            # Process predictions and generate prefetches
            candidates = tf.argmax(candidate_logits, axis=1).numpy()
            offsets_pred = tf.argmax(offset_logits, axis=1).numpy()
            
            # Process each prediction in the mini-batch
            for i in range(current_batch_size):
                is_debug_sample = i in debug_sample_indices
                
                # Debug output
                if is_debug_sample and debug_f:
                    debug_f.write(f"\n===== Debug Sample {line_count - current_batch_size + i} =====\n")
                    debug_f.write(f"PC: {pcs_batch[i]:x}, Address: {addrs_batch[i]:x}\n")
                    debug_f.write(f"Page: {pages_batch[i]:x}, Offset: {offsets_batch[i]}\n")
                    debug_f.write(f"Cluster History: {cluster_histories_batch[i]}\n")
                    debug_f.write(f"Offset History: {offset_histories_batch[i]}\n")
                    debug_f.write(f"Prediction - Candidate: {candidates[i]}, Offset: {offsets_pred[i]}\n")
                
                # Check if "no prefetch" was predicted
                if candidates[i] == config.num_candidates:
                    stats['no_prefetch_predictions'] += 1
                    stats['total_accesses'] += 1
                    continue
                
                stats['model_predictions'] += 1
                
                # Get trigger cluster and page
                trigger_cluster = cluster_histories_batch[i][-1]
                trigger_page = pages_batch[i]
                
                # Get candidate pages
                candidate_pages = prefetcher.metadata_manager.get_candidate_pages(
                    cluster_histories_batch[i][-1]  # Use last page in history
                )
                
                # Determine prefetch page and address
                prefetch_addr = None
                
                # Method 1: Try to use candidate_pages from metadata
                if candidate_pages and candidates[i] < len(candidate_pages):
                    prefetch_page = candidate_pages[candidates[i]][0]
                    prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (offsets_pred[i] << 6)
                    stats['valid_prefetches'] += 1
                    
                    if is_debug_sample and debug_f:
                        debug_f.write(f"Using metadata candidate: Page {prefetch_page:x}\n")
                
                # Method 2: If Method 1 fails, try to use cluster_to_pages mapping
                elif cluster_to_pages and trigger_cluster in cluster_to_pages and cluster_to_pages[trigger_cluster]:
                    stats['empty_candidate_pages'] += 1
                    
                    # Select representative page from this cluster
                    prefetch_page = cluster_to_pages[trigger_cluster][0]
                    prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (offsets_pred[i] << 6)
                    
                    if is_debug_sample and debug_f:
                        debug_f.write(f"Using cluster mapping: Cluster {trigger_cluster} -> Page {prefetch_page:x}\n")
                
                # Method 3: Fallback to current page with predicted offset
                else:
                    stats['fallback_to_current_page'] += 1
                    
                    if not candidate_pages:
                        stats['empty_candidate_pages'] += 1
                    else:
                        stats['candidate_index_out_of_range'] += 1
                    
                    prefetch_page = trigger_page
                    prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (offsets_pred[i] << 6)
                    
                    if is_debug_sample and debug_f:
                        debug_f.write(f"Falling back to current page\n")
                
                # Check if prefetch equals current address
                if prefetch_addr == addrs_batch[i]:
                    # Use next cache line instead
                    prefetch_addr = addrs_batch[i] + 64
                    stats['next_line_prefetches'] += 1
                    
                    if is_debug_sample and debug_f:
                        debug_f.write(f"Using next line instead: {prefetch_addr:x}\n")
                
                # Apply prefetch distance if specified
                trigger_id = inst_ids_batch[i]
                if prefetch_distance > 0:
                    trigger_id = max(0, inst_ids_batch[i] - prefetch_distance)
                
                # Write prefetch to file
                out_f.write(f"{trigger_id} {prefetch_addr:x}\n")
                
                if is_debug_sample and debug_f:
                    debug_f.write(f"Final output: {trigger_id} {prefetch_addr:x}\n")
                    debug_f.write("================================\n")
                
                stats['total_accesses'] += 1
            
            # Print progress periodically
            if line_count % 100000 == 0:
                elapsed = time.time() - start_time
                stats['processing_time'] = elapsed
                
                progress = (line_count / total_lines) * 100
                rate = line_count / max(1, elapsed)
                remaining = (total_lines - line_count) / max(1, rate)
                
                print(f"Progress: {line_count}/{total_lines} ({progress:.2f}%) - {rate:.2f} lines/sec, Est. remaining: {remaining/60:.2f} min")
                
                # Print metadata statistics periodically
                if line_count % 1000000 == 0:
                    print(f"Metadata size: {prefetcher.metadata_manager.get_metadata_size_kb():.2f} KB")
                    print(f"Pages with metadata: {len(prefetcher.metadata_manager.page_metadata)}")
    
    # Close debug file if opened
    if debug_f:
        debug_f.close()
    
    # Print final statistics
    elapsed = time.time() - start_time
    stats['processing_time'] = elapsed
    
    print("\n=== Final Statistics ===")
    print(f"Total Lines Processed: {line_count}")
    print(f"Processing Time: {elapsed:.2f} seconds ({line_count/elapsed:.2f} lines/sec)")
    print(f"Prediction Time: {stats['prediction_time']:.2f} seconds ({stats['prediction_time']/line_count*1000:.2f} ms/prediction)")
    
    # Print detailed statistics
    print(f"\nPrefetch Statistics:")
    print(f"Total Accesses: {stats['total_accesses']}")
    print(f"Model Predictions: {stats['model_predictions']} ({stats['model_predictions']/stats['total_accesses']*100:.2f}%)")
    print(f"No Prefetch Predictions: {stats['no_prefetch_predictions']} ({stats['no_prefetch_predictions']/stats['total_accesses']*100:.2f}%)")
    
    if stats['model_predictions'] > 0:
        print(f"Empty Candidate Pages: {stats['empty_candidate_pages']} ({stats['empty_candidate_pages']/stats['model_predictions']*100:.2f}%)")
        print(f"Candidate Index Out of Range: {stats['candidate_index_out_of_range']} ({stats['candidate_index_out_of_range']/stats['model_predictions']*100:.2f}%)")
        print(f"Fallback to Current Page: {stats['fallback_to_current_page']} ({stats['fallback_to_current_page']/stats['model_predictions']*100:.2f}%)")
        print(f"Valid Prefetches: {stats['valid_prefetches']} ({stats['valid_prefetches']/stats['model_predictions']*100:.2f}%)")
        print(f"Next Line Prefetches: {stats['next_line_prefetches']} ({stats['next_line_prefetches']/stats['model_predictions']*100:.2f}%)")
    
    return stats

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

    # Print random sample of mappings
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
    
    # Run clustering tests if requested
    if args.test_clustering:
        test_clustering_mapping(clustering_info, config)
    
    # Generate prefetches
    if args.sequential:
        # Use sequential processing (recommended)
        stats = generate_prefetches_sequential(
            model=model,
            clustering_info=clustering_info,
            trace_path=args.benchmark,
            config=config,
            output_path=args.output,
            prefetch_distance=args.prefetch_distance,
            debug_file=args.debug_file
        )
    elif args.simple_mode:
        # Use the original simple mode
        from your_simple_mode_function import generate_prefetches_simple
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
        # Use the original batch processing mode
        from your_batch_function import generate_prefetches
        generate_prefetches(
            model=model,
            clustering_info=clustering_info,
            trace_path=args.benchmark,
            config=config,
            output_path=args.output,
            prefetch_distance=args.prefetch_distance,
            debug_file=args.debug_file
        )
    
    # Generate multiple prefetch files with different distances if requested
    if args.prefetch_distance == 0:
        print("\n=== Generating Multiple Prefetch Files with Different Distances ===")
        distances = [20, 50, 100, 200, 500]
        
        for distance in distances:
            distance_output = f"{os.path.splitext(args.output)[0]}_d{distance}{os.path.splitext(args.output)[1]}"
            distance_debug = None
            if args.debug_file:
                distance_debug = f"{os.path.splitext(args.debug_file)[0]}_d{distance}.log"
            
            print(f"\nGenerating prefetch file with distance {distance}...")
            
            # Use sequential processing (recommended)
            if args.sequential:
                generate_prefetches_sequential(
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