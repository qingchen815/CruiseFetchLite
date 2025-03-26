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
    Generate prefetch file using T-LITE model
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
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
            
            # Update prefetcher history
            prefetcher.update_history(page, offset, pc)
            
            # Generate prefetches
            prefetches = prefetcher.get_prefetches()
            
            # Write prefetches to file
            if prefetches:
                out_f.write(f"{inst_id}:")
                for i, (candidate, offset) in enumerate(prefetches):
                    if i > 0:
                        out_f.write(",")
                    out_f.write(f"{candidate}:{offset}")
                out_f.write("\n")
    
    f.close()
    print(f"Generated prefetch file: {output_path}")

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    model = create_tlite_model(config)
    model.load_weights(args.model_path)
    
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