import argparse
import numpy as np
import tensorflow as tf
import lzma
import os

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
    Generate prefetch file using T-LITE model with batched processing
    """
    print(f"Generating prefetches for {trace_path}")
    
    # Initialize prefetcher
    prefetcher = TLITEPrefetcher(
        model=model,
        clustering_info=clustering_info,
        config=config
    )
    
    # Read trace file
    if trace_path.endswith('.txt.xz'):
        f = lzma.open(trace_path, mode='rt', encoding='utf-8')
    else:
        f = open(trace_path, 'r')
    
    # Count total lines for progress reporting
    print("Counting total lines...")
    total_lines = sum(1 for line in f if not (line.startswith('***') or line.startswith('Read')))
    f.seek(0)
    print(f"Total lines to process: {total_lines}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Batch processing parameters
    batch_size = 1000
    current_batch = []
    batch_metadata = []  # Store inst_id and other info for output
    processed_lines = 0
    
    # Process trace and generate prefetches
    with open(output_path, 'w') as out_f:
        for line in f:
            # Skip comments
            if line.startswith('***') or line.startswith('Read'):
                continue
            
            # Parse line
            split = line.strip().split(', ')
            inst_id = int(split[0])
            pc = int(split[3], 16)
            addr = int(split[2], 16)
            
            # Convert address to page and offset
            page = (addr >> 6) >> config.offset_bits
            offset = (addr >> 6) & ((1 << config.offset_bits) - 1)
            
            # Add to current batch
            current_batch.append((page, offset, pc))
            batch_metadata.append(inst_id)
            
            # Process batch when full or at end of file
            if len(current_batch) >= batch_size:
                process_batch(prefetcher, current_batch, batch_metadata, out_f)
                current_batch = []
                batch_metadata = []
                
                # Update progress
                processed_lines += batch_size
                if processed_lines % 10000 == 0:
                    print(f"Progress: {processed_lines}/{total_lines} lines ({processed_lines/total_lines*100:.2f}%)")
        
        # Process remaining items in the last batch
        if current_batch:
            process_batch(prefetcher, current_batch, batch_metadata, out_f)
            processed_lines += len(current_batch)
    
    f.close()
    print(f"Generated prefetch file: {output_path}")
    print(f"Processed {processed_lines} lines total")
    prefetcher.print_stats()  # Print final statistics

def process_batch(prefetcher, batch, batch_metadata, out_file):
    """
    Process a batch of memory accesses
    
    Args:
        prefetcher: TLITEPrefetcher instance
        batch: List of (page, offset, pc) tuples
        batch_metadata: List of instruction IDs
        out_file: Output file handle
    """
    for (page, offset, pc), inst_id in zip(batch, batch_metadata):
        # Update history and get prefetches
        prefetcher.update_history(page, offset, pc)
        prefetches = prefetcher.get_prefetches()
        
        # Write prefetches to file
        if prefetches:
            out_file.write(f"{inst_id}:")
            for i, (candidate, offset) in enumerate(prefetches):
                if i > 0:
                    out_file.write(",")
                out_file.write(f"{candidate}:{offset}")
            out_file.write("\n")

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model, ignore warnings
    model = create_tlite_model(config)
    model.load_weights(args.model_path).expect_partial()
    
    # Load clustering information
    clustering_info = np.load(args.clustering_path, allow_pickle=True).item()
    
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