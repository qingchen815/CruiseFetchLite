import argparse
import numpy as np
import tensorflow as tf
import lzma
import os
import time
from collections import defaultdict, deque

class HybridStreamProcessor:
    """
    Processes memory trace with hybrid stream/window approach that
    both preserves access order and enables fast processing
    """
    def __init__(self, prefetchers, config, num_streams=32, window_size=1000):
        self.prefetchers = prefetchers
        self.config = config
        self.num_streams = num_streams
        self.window_size = window_size
        
        # Metadata tracking
        self.stream_windows = [deque(maxlen=window_size) for _ in range(num_streams)]
        self.stream_stats = {i: {'updates': 0, 'prefetches': 0} for i in range(num_streams)}
        
        # Page continuity tracking (maps page to last stream that accessed it)
        self.page_stream_map = {}
        
        # Recent page transitions per stream
        self.stream_transitions = [defaultdict(dict) for _ in range(num_streams)]
        
        # Global statistics
        self.stats = {
            'total_accesses': 0,
            'metadata_updates': 0,
            'empty_candidates': 0,
            'valid_prefetches': 0,
            'next_line_prefetches': 0,
            'fallback_prefetches': 0
        }
    
    def calculate_stream_id(self, pc, addr, page):
        """
        Calculate stream ID with page continuity - prioritizes previous stream assignment
        for the same page to maintain access pattern integrity
        """
        # Check if this page was previously assigned to a stream
        if page in self.page_stream_map:
            return self.page_stream_map[page]
        
        # Otherwise calculate using hash function
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
        
        return hash_value % self.num_streams
    
    def process_access(self, inst_id, pc, addr, output_file=None):
        """Process a single memory access, update metadata and generate prefetch"""
        # Extract page and offset
        page = (addr >> 6) >> self.config.offset_bits
        offset = (addr >> 6) & ((1 << self.config.offset_bits) - 1)
        
        # Calculate stream ID
        stream_id = self.calculate_stream_id(pc, addr, page)
        
        # Update page-to-stream mapping for continuity
        self.page_stream_map[page] = stream_id
        
        # Get prefetcher for this stream
        prefetcher = self.prefetchers[stream_id]
        
        # Update statistics
        self.stats['total_accesses'] += 1
        self.stream_stats[stream_id]['updates'] += 1
        
        # Add to stream window
        self.stream_windows[stream_id].append((inst_id, pc, page, offset))
        
        # Process stream window for metadata updates
        self._process_stream_window(stream_id)
        
        # Generate prefetch if requested
        if output_file:
            prefetch = self._generate_prefetch(stream_id, inst_id, pc, page, offset)
            if prefetch:
                output_file.write(f"{inst_id} {prefetch:x}\n")
                self.stream_stats[stream_id]['prefetches'] += 1
    
    def _process_stream_window(self, stream_id):
        """Process all accesses in a stream window to update metadata in order"""
        window = self.stream_windows[stream_id]
        prefetcher = self.prefetchers[stream_id]
        
        # Need at least 2 accesses to establish a relationship
        if len(window) < 2:
            return
            
        # Update metadata for the most recent transition (last two accesses)
        second_last_idx = len(window) - 2
        last_idx = len(window) - 1
        
        # Extract data
        _, _, prev_page, prev_offset = window[second_last_idx]
        _, curr_pc, curr_page, curr_offset = window[last_idx]
        
        # Update metadata
        prefetcher.metadata_manager.update_page_access(
            prev_page, curr_page, prev_offset, curr_offset)
        self.stats['metadata_updates'] += 1
        
        # Also store in stream transitions for quick lookup
        if prev_page not in self.stream_transitions[stream_id]:
            self.stream_transitions[stream_id][prev_page] = {}
        
        # Count transition frequency
        if curr_page in self.stream_transitions[stream_id][prev_page]:
            self.stream_transitions[stream_id][prev_page][curr_page] += 1
        else:
            self.stream_transitions[stream_id][prev_page][curr_page] = 1
    
    def _generate_prefetch(self, stream_id, inst_id, pc, page, offset):
        """Generate a prefetch for the given access"""
        prefetcher = self.prefetchers[stream_id]
        
        # Check if we have candidate pages from metadata
        candidates = prefetcher.metadata_manager.get_candidate_pages(page)
        
        # If no candidates from metadata, check stream transitions
        if not candidates and page in self.stream_transitions[stream_id]:
            # Create candidates from stream transitions
            transitions = self.stream_transitions[stream_id][page]
            candidates = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
            candidates = candidates[:self.config.num_candidates]  # Limit to N candidates
        
        if not candidates:
            self.stats['empty_candidates'] += 1
            
            # Fallback: Try to predict next line within same page
            next_offset = (offset + 1) % 64
            prefetch_addr = (page << (self.config.offset_bits + 6)) | (next_offset << 6)
            self.stats['next_line_prefetches'] += 1
            return prefetch_addr
        
        # Get candidate prediction from neural model
        # Prepare model inputs
        cluster_id = prefetcher.mapping.get(page, 0)  # Get cluster ID for this page
        
        # Make prediction using the neural model
        candidate_logits, offset_logits = prefetcher.model((
            np.array([[cluster_id]]),
            np.array([[offset]]),
            np.array([[pc]]),
            np.array([[[1.0] * self.config.num_candidates]])  # Dummy DPF
        ))
        
        # Get predictions
        candidate_idx = np.argmax(candidate_logits[0])
        offset_pred = np.argmax(offset_logits[0])
        
        # Check if "no prefetch" was predicted
        if candidate_idx == self.config.num_candidates:
            # Fallback to next line prefetch
            next_offset = (offset + 1) % 64
            prefetch_addr = (page << (self.config.offset_bits + 6)) | (next_offset << 6)
            self.stats['next_line_prefetches'] += 1
            return prefetch_addr
        
        # Get prefetch page
        if candidate_idx < len(candidates):
            prefetch_page = candidates[candidate_idx][0]
            self.stats['valid_prefetches'] += 1
        else:
            # Fallback to current page
            prefetch_page = page
            self.stats['fallback_prefetches'] += 1
        
        # Construct prefetch address
        prefetch_addr = (prefetch_page << (self.config.offset_bits + 6)) | (offset_pred << 6)
        return prefetch_addr
    
    def process_file(self, trace_path, output_path):
        """Process an entire trace file"""
        print(f"Processing trace file: {trace_path}")
        print(f"Output file: {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Track timing
        start_time = time.time()
        processed_lines = 0
        last_print_time = start_time
        
        # Process file
        if trace_path.endswith('.txt.xz'):
            f = lzma.open(trace_path, mode='rt', encoding='utf-8')
        else:
            f = open(trace_path, 'r')
        
        with open(output_path, 'w', buffering=1024*1024) as out_f:
            for line in f:
                if line.startswith('***') or line.startswith('Read'):
                    continue
                
                # Parse line
                split = line.strip().split(', ')
                if len(split) < 4:
                    continue
                
                inst_id = int(split[0])
                pc = int(split[3], 16)
                addr = int(split[2], 16)
                
                # Process this access
                self.process_access(inst_id, pc, addr, out_f)
                
                processed_lines += 1
                
                # Print progress periodically
                current_time = time.time()
                if current_time - last_print_time >= 10:  # Every 10 seconds
                    print(f"Processed {processed_lines} lines ({processed_lines/(current_time-start_time):.2f} lines/sec)")
                    last_print_time = current_time
        
        f.close()
        
        # Print final statistics
        processing_time = time.time() - start_time
        print(f"\n=== Processing Complete ===")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"Processing rate: {processed_lines/processing_time:.2f} lines/second")
        self.print_statistics()
    
    def print_statistics(self):
        """Print detailed statistics about the generation process"""
        print("\n=== Detailed Statistics ===")
        print(f"Total Accesses: {self.stats['total_accesses']}")
        print(f"Metadata Updates: {self.stats['metadata_updates']} ({self.stats['metadata_updates']/max(1, self.stats['total_accesses'])*100:.2f}%)")
        
        total_prefetches = (self.stats['valid_prefetches'] + 
                           self.stats['next_line_prefetches'] + 
                           self.stats['fallback_prefetches'])
        
        print(f"\nPrefetch Generation:")
        print(f"  Total Prefetches: {total_prefetches} ({total_prefetches/max(1, self.stats['total_accesses'])*100:.2f}%)")
        print(f"  Valid Prefetches: {self.stats['valid_prefetches']} ({self.stats['valid_prefetches']/max(1, total_prefetches)*100:.2f}%)")
        print(f"  Next Line Prefetches: {self.stats['next_line_prefetches']} ({self.stats['next_line_prefetches']/max(1, total_prefetches)*100:.2f}%)")
        print(f"  Fallback Prefetches: {self.stats['fallback_prefetches']} ({self.stats['fallback_prefetches']/max(1, total_prefetches)*100:.2f}%)")
        print(f"  Empty Candidate Lists: {self.stats['empty_candidates']} ({self.stats['empty_candidates']/max(1, self.stats['total_accesses'])*100:.2f}%)")
        
        # Print stream distribution
        print("\n=== Stream Distribution ===")
        total_updates = sum(stats['updates'] for stats in self.stream_stats.values())
        if total_updates > 0:
            active_streams = [(sid, stats) for sid, stats in self.stream_stats.items() 
                           if stats['updates'] > 0]
            active_streams.sort(key=lambda x: x[1]['updates'], reverse=True)
            
            print(f"Active Streams: {len(active_streams)}/{self.num_streams}")
            print("\nTop 5 Most Active Streams:")
            for stream_id, stats in active_streams[:5]:
                prefetch_rate = (stats['prefetches'] / stats['updates'] * 100 
                               if stats['updates'] > 0 else 0)
                print(f"Stream {stream_id}:")
                print(f"   Load: {stats['updates']/total_updates*100:.2f}% ({stats['updates']} updates)")
                print(f"   Prefetches: {prefetch_rate:.2f}% ({stats['prefetches']} prefetches)")


def generate_prefetches_hybrid(model, clustering_info, trace_path, config, output_path, num_streams=32):
    """
    Generate prefetch file using T-LITE model with hybrid stream/window processing
    for better metadata management while maintaining speed
    """
    # Create mapping from pages to clusters
    page_to_cluster = clustering_info.get('page_to_cluster', {})
    print(f"Loaded {len(page_to_cluster)} page-to-cluster mappings")
    
    # Create prefetchers (one per stream)
    prefetchers = []
    for i in range(num_streams):
        prefetcher = PreFetcherWrapper(
            model=model,
            mapping=page_to_cluster,
            config=config
        )
        prefetchers.append(prefetcher)
    
    # Create processor
    processor = HybridStreamProcessor(
        prefetchers=prefetchers,
        config=config,
        num_streams=num_streams,
        window_size=1000  # Keep track of 1000 accesses per stream
    )
    
    # Process file
    processor.process_file(trace_path, output_path)


class PreFetcherWrapper:
    """
    Simplified wrapper around the model and metadata manager
    focused on fast prefetch generation
    """
    def __init__(self, model, mapping, config):
        self.model = model
        self.mapping = mapping
        self.config = config
        self.metadata_manager = MetadataManager(num_candidates=config.num_candidates)
        

class MetadataManager:
    """
    Simplified metadata manager focused on fast updates and candidate retrieval
    """
    def __init__(self, num_candidates=4):
        self.num_candidates = num_candidates
        self.page_metadata = {}  # Maps page_id -> {successors: {successor_page: count}}
    
    def update_page_access(self, trigger_page, next_page, trigger_offset, next_offset):
        """Update metadata with a page transition"""
        # Initialize metadata for trigger page if not exists
        if trigger_page not in self.page_metadata:
            self.page_metadata[trigger_page] = {'successors': {}}
        
        # Update successor frequency
        successors = self.page_metadata[trigger_page]['successors']
        if next_page in successors:
            successors[next_page] += 1
        else:
            successors[next_page] = 1
    
    def get_candidate_pages(self, trigger_page):
        """Get the top N candidate pages for a trigger page"""
        if trigger_page not in self.page_metadata:
            return []
        
        # Get successors and sort by frequency
        successors = self.page_metadata[trigger_page]['successors']
        sorted_successors = sorted(successors.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N candidates
        return sorted_successors[:self.num_candidates]


def main():
    parser = argparse.ArgumentParser(description='Generate prefetch files using T-LITE model with improved metadata handling')
    parser.add_argument('--model-path', help='Path to model checkpoint', required=True)
    parser.add_argument('--clustering-path', help='Path to clustering information', required=True)
    parser.add_argument('--benchmark', help='Path to benchmark trace', required=True)
    parser.add_argument('--output', help='Path to output prefetch file', required=True)
    parser.add_argument('--config', default='./config/TLITE2debug1.yaml', help='Path to configuration file')
    parser.add_argument('--num-streams', type=int, default=32, help='Number of streams for parallel processing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    print("\n=== Loading Model ===")
    model = create_tlite_model(config)
    model.load_weights(args.model_path).expect_partial()
    
    # Load clustering information
    print("\n=== Loading Clustering Information ===")
    clustering_info = np.load(args.clustering_path, allow_pickle=True).item()
    
    # Generate prefetches using hybrid method
    generate_prefetches_hybrid(
        model=model,
        clustering_info=clustering_info,
        trace_path=args.benchmark,
        config=config,
        output_path=args.output,
        num_streams=args.num_streams
    )
    
    print("Prefetch generation complete!")


if __name__ == "__main__":
    main()