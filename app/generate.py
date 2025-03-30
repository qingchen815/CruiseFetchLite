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
    parser.add_argument('--config', default='./config/TLITE2debug1.yaml', help='Path to configuration file')
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

def generate_prefetches_sequential(model, clustering_info, trace_path, config, output_path, prefetch_distance=0, debug_file=None):
    """
    Generate prefetch file using T-LITE model with sequential processing logic
    that maintains the correct access order while remaining efficient
    """
    print(f"\n=== Starting Sequential Prefetch Generation ===")
    print(f"Processing trace file: {trace_path}")
    print(f"Output file: {output_path}")
    print(f"Prefetch distance: {prefetch_distance}")
    
    # Create debug file handler if specified
    debug_f = None
    if debug_file:
        os.makedirs(os.path.dirname(debug_file), exist_ok=True)
        debug_f = open(debug_file, 'w')
        print(f"Debug info will be written to: {debug_file}")
    
    # Test clustering mapping and get cluster_to_pages
    cluster_to_pages = test_clustering_mapping(clustering_info, config)
    
    # Get page to cluster mapping
    page_to_cluster = clustering_info.get('page_to_cluster', {})
    
    # Create the prefetcher
    prefetcher = TLITEPrefetcher(
        model=model,
        clustering_info=clustering_info,
        config=config
    )
    
    # Create metadata manager validator to check its state
    metadata_manager = prefetcher.metadata_manager
    
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
    
    mapper = DynamicClusterMapper(page_to_cluster, config.num_clusters)
    
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
    
    # Statistics
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
    
    # Create debug sample set
    debug_samples = set()
    
    # Process trace file line by line
    line_count = 0
    start_time = time.time()
    
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
            is_debug_sample = (line_count % 10000 == 0) or (len(debug_samples) < 100 and random.random() < 0.005)
            if is_debug_sample:
                debug_samples.add(inst_id)
                
                if debug_f:
                    debug_f.write(f"\n===== Debug Sample {inst_id} =====\n")
                    debug_f.write(f"Line: {line.strip()}\n")
                    debug_f.write(f"PC: {pc:x}, Address: {addr:x}\n")
                    debug_f.write(f"Page: {page:x}, Offset: {offset}\n")
                    debug_f.write(f"Current History - Pages: {prefetcher.page_history}, Offsets: {prefetcher.offset_history}\n")
            
            # Get cluster for current page
            cluster_id = mapper.get_cluster(page)
            
            # Update prefetcher state with this access - do this BEFORE prediction to ensure history is current
            prefetcher.update_history(cluster_id, offset, pc)
            
            # Make prediction
            prediction_start = time.time()
            
            # Prepare model inputs
            cluster_history, offset_history, _, dpf_vectors = prefetcher.metadata_manager.prepare_model_inputs(
                prefetcher.page_history,
                prefetcher.offset_history,
                pc,
                prefetcher.cluster_mapping_utils
            )
            
            # Convert to tensors
            cluster_input = tf.convert_to_tensor([cluster_history], dtype=tf.int32)
            offset_input = tf.convert_to_tensor([offset_history], dtype=tf.int32)
            pc_input = tf.convert_to_tensor([[pc]], dtype=tf.int32)
            dpf_input = tf.convert_to_tensor([dpf_vectors], dtype=tf.float32)
            
            # Make prediction
            candidate_logits, offset_logits = model((cluster_input, offset_input, pc_input, dpf_input))
            
            # Get predicted candidate and offset
            candidate_pred = tf.argmax(candidate_logits[0]).numpy()
            offset_pred = tf.argmax(offset_logits[0]).numpy()
            
            stats['prediction_time'] += time.time() - prediction_start
            
            # Debug prediction
            if is_debug_sample and debug_f:
                debug_f.write(f"Model Inputs:\n")
                debug_f.write(f"  Cluster History: {cluster_history}\n")
                debug_f.write(f"  Offset History: {offset_history}\n")
                debug_f.write(f"  PC: {pc:x}\n")
                debug_f.write(f"  DPF Vectors: {dpf_vectors}\n")
                debug_f.write(f"Prediction Results:\n")
                debug_f.write(f"  Candidate Prediction: {candidate_pred}\n")
                debug_f.write(f"  Offset Prediction: {offset_pred}\n")
                debug_f.write(f"  Candidate Logits: {candidate_logits[0].numpy()}\n")
                debug_f.write(f"  Offset Logits: {offset_logits[0].numpy()}\n")
            
            # Check if "no prefetch" was predicted
            if candidate_pred == config.num_candidates:
                stats['no_prefetch_predictions'] += 1
                if is_debug_sample and debug_f:
                    debug_f.write("No prefetch predicted\n")
                
                # Update statistics and continue
                stats['total_accesses'] += 1
                line_count += 1
                continue
            
            stats['model_predictions'] += 1
            
            # Get the trigger cluster ID (last cluster in history)
            trigger_cluster = cluster_history[-1]
            
            # Get candidate pages for this cluster
            candidate_pages = prefetcher.metadata_manager.get_candidate_pages(prefetcher.page_history[-1])
            
            if is_debug_sample and debug_f:
                debug_f.write(f"Trigger Cluster: {trigger_cluster}\n")
                debug_f.write(f"Candidate Pages: {candidate_pages}\n")
                
                # Check DPF metadata directly
                debug_f.write(f"Metadata for page {prefetcher.page_history[-1]:x}:\n")
                if prefetcher.page_history[-1] in metadata_manager.page_metadata:
                    meta = metadata_manager.page_metadata[prefetcher.page_history[-1]]
                    debug_f.write(f"  Successors: {meta['successors']}\n")
                    sum_trans = np.sum(meta['offset_transitions'])
                    if sum_trans > 0:
                        debug_f.write(f"  Offset Transitions Sum: {sum_trans}\n")
                    else:
                        debug_f.write("  Offset Transitions: Empty (all zeros)\n")
                else:
                    debug_f.write("  No metadata found for this page\n")
            
            # Determine prefetch page and address
            prefetch_addr = None
            
            # Method 1: Try to use candidate_pages from metadata
            if candidate_pages and candidate_pred < len(candidate_pages):
                prefetch_page = candidate_pages[candidate_pred][0]
                prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (offset_pred << 6)
                stats['valid_prefetches'] += 1
                
                if is_debug_sample and debug_f:
                    debug_f.write(f"Using metadata candidate: Page {prefetch_page:x} with offset {offset_pred}\n")
                    debug_f.write(f"Prefetch address: {prefetch_addr:x}\n")
            
            # Method 2: If Method 1 fails, try to use cluster_to_pages mapping
            elif cluster_to_pages and trigger_cluster in cluster_to_pages and cluster_to_pages[trigger_cluster]:
                stats['empty_candidate_pages'] += 1
                
                # Select representative page from this cluster
                prefetch_page = cluster_to_pages[trigger_cluster][0]
                prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (offset_pred << 6)
                
                if is_debug_sample and debug_f:
                    debug_f.write(f"Using cluster_to_pages mapping: Cluster {trigger_cluster} -> Page {prefetch_page:x}\n")
                    debug_f.write(f"Prefetch address: {prefetch_addr:x}\n")
            
            # Method 3: Fallback to current page with predicted offset
            else:
                stats['fallback_to_current_page'] += 1
                
                if not candidate_pages:
                    stats['empty_candidate_pages'] += 1
                else:
                    stats['candidate_index_out_of_range'] += 1
                
                prefetch_page = page
                prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (offset_pred << 6)
                
                if is_debug_sample and debug_f:
                    debug_f.write(f"Falling back to current page with predicted offset\n")
                    debug_f.write(f"Prefetch address: {prefetch_addr:x}\n")
            
            # Check if prefetch equals current address
            if prefetch_addr == addr:
                # Use next cache line instead
                prefetch_addr = addr + 64
                stats['next_line_prefetches'] += 1
                
                if is_debug_sample and debug_f:
                    debug_f.write(f"Prefetch equals current address, using next line: {prefetch_addr:x}\n")
            
            # Apply prefetch distance if specified
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
            
            # Update statistics and progress
            stats['total_accesses'] += 1
            line_count += 1
            
            # Update metadata with this access
            if is_debug_sample and debug_f:
                debug_f.write(f"Next access metadata update:\n")
                debug_f.write(f"  Previous page: {prefetcher.page_history[-2]:x}, Previous offset: {prefetcher.offset_history[-2]}\n")
                debug_f.write(f"  Current page: {prefetcher.page_history[-1]:x}, Current offset: {prefetcher.offset_history[-1]}\n")
            
            # Print progress periodically
            if line_count % 100000 == 0:
                elapsed = time.time() - start_time
                stats['processing_time'] = elapsed
                
                progress = (line_count / total_lines) * 100
                rate = line_count / max(1, elapsed)
                remaining = (total_lines - line_count) / max(1, rate)
                
                print(f"Progress: {line_count}/{total_lines} ({progress:.2f}%) - {rate:.2f} lines/sec, Est. remaining: {remaining/60:.2f} min")
                
                # Print metadata statistics 
                if line_count % 1000000 == 0:
                    print(f"Metadata size: {metadata_manager.get_metadata_size_kb():.2f} KB")
                    print(f"Pages with metadata: {len(metadata_manager.page_metadata)}")
                    
                    # Check a few random pages for debugging
                    if metadata_manager.page_metadata:
                        debug_pages = random.sample(list(metadata_manager.page_metadata.keys()), 
                                                  min(3, len(metadata_manager.page_metadata)))
                        
                        for dp in debug_pages:
                            meta = metadata_manager.page_metadata[dp]
                            successors = len(meta['successors'])
                            print(f"  Page {dp:x}: {successors} successors, total transitions: {np.sum(meta['offset_transitions'])}")
    
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
    
    # Print mapper statistics
    mapper.print_stats()
    
    # Print metadata manager statistics
    print("\nMetadata Manager Statistics:")
    print(f"Metadata Size: {metadata_manager.get_metadata_size_kb():.2f} KB")
    print(f"Pages with Metadata: {len(metadata_manager.page_metadata)}")
    
    # Check for empty metadata entries
    if metadata_manager.page_metadata:
        empty_succs = sum(1 for meta in metadata_manager.page_metadata.values() if not meta['successors'])
        empty_trans = sum(1 for meta in metadata_manager.page_metadata.values() if np.sum(meta['offset_transitions']) == 0)
        
        print(f"Pages with Empty Successors: {empty_succs} ({empty_succs/len(metadata_manager.page_metadata)*100:.2f}%)")
        print(f"Pages with Empty Transitions: {empty_trans} ({empty_trans/len(metadata_manager.page_metadata)*100:.2f}%)")
        
        # Check distribution of successor counts
        succ_counts = [len(meta['successors']) for meta in metadata_manager.page_metadata.values()]
        if succ_counts:
            avg_succs = sum(succ_counts) / len(succ_counts)
            max_succs = max(succ_counts) if succ_counts else 0
            min_succs = min(succ_counts) if succ_counts else 0
            
            print(f"Successor Statistics:")
            print(f"  Average Successors per Page: {avg_succs:.2f}")
            print(f"  Max Successors for a Page: {max_succs}")
            print(f"  Min Successors for a Page: {min_succs}")
    
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