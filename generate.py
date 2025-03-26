import argparse
import numpy as np
import tensorflow as tf
import lzma
import os

from tlite.config import ModelConfig, extend_voyager_config_for_tlite
from tlite.models import create_tlite_model
from tlite.prefetcher import TLITEPrefetcher
from tlite.clustering import BehavioralClusteringUtils

def parse_args():
    parser = argparse.ArgumentParser(description='Generate prefetch file for ChampSim')
    parser.add_argument('--benchmark', help='Path to the benchmark trace', required=True)
    parser.add_argument('--model-path', help='Path to trained model', required=True)
    parser.add_argument('--prefetch-file', help='Path to output prefetch file', required=True)
    parser.add_argument('--config', default='./configs/tlite.yaml', help='Path to configuration file')
    parser.add_argument('--train', action='store_true', default=False, help='Generate for train dataset')
    parser.add_argument('--valid', action='store_true', default=False, help='Generate for valid dataset')
    parser.add_argument('--no-test', action='store_true', default=False, help='Skip test dataset')
    parser.add_argument('--twilight', action='store_true', default=False, help='Use Twilight model instead of T-LITE')
    return parser.parse_args()

def create_prefetch_file(prefetch_file, inst_ids, addresses, append=False):
    '''Create a prefetch file for ChampSim'''
    with open(prefetch_file, 'a' if append else 'w') as f:
        for inst_id, addr in zip(inst_ids, addresses):
            print(inst_id, hex(addr), file=f)

def generate_prefetches(trace_path, model, config, twilight=False):
    """Generate prefetches for a trace file"""
    print(f"Generating prefetches for: {trace_path}")
    
    # Create prefetcher
    prefetcher = TLITEPrefetcher(model, config)
    
    # Load clustering information if using T-LITE
    if not twilight:
        clustering_path = os.path.join(os.path.dirname(config.model_path), 'clustering.npy')
        if os.path.exists(clustering_path):
            clustering = np.load(clustering_path, allow_pickle=True).item()
            prefetcher.metadata_manager.page_cluster_map = clustering['cluster_map']
    
    # Read trace file
    if trace_path.endswith('.txt.xz'):
        f = lzma.open(trace_path, mode='rt', encoding='utf-8')
    else:
        f = open(trace_path, 'r')
    
    # Track prefetches
    prefetch_addresses = []
    inst_ids = []
    
    # Process line by line
    for line in f:
        # Skip comments
        if line.startswith('***') or line.startswith('Read'):
            continue
            
        # Parse line
        split = line.strip().split(', ')
        inst_id = int(split[0])
        pc = int(split[3], 16)
        addr = int(split[2], 16)
        
        # Get prefetch
        prefetch = prefetcher.handle_access(addr, pc)
        
        # Store if a prefetch was generated
        if prefetch is not None:
            prefetch_addresses.append(prefetch)
            inst_ids.append(inst_id)
    
    f.close()
    
    print(f"Generated {len(prefetch_addresses)} prefetches")
    prefetcher.print_stats()
    
    return inst_ids, prefetch_addresses

def main():
    args = parse_args()
    
    # Load configuration
    config = extend_voyager_config_for_tlite(args.config)
    
    # Set model path
    config.model_path = args.model_path
    
    # Create model
    model = create_tlite_model(config)
    
    # Load model weights
    model.load_weights(args.model_path)
    
    # Determine which parts of the trace to process
    start_test = 0
    if args.train:
        start_test = 200_000_000  # Skip first 200M instructions
    if args.valid:
        start_test = 225_000_000  # Skip first 225M instructions
    
    # Generate prefetches
    inst_ids = []
    prefetch_addresses = []
    
    # Read trace and generate prefetches
    if args.train or args.valid or not args.no_test:
        trace_ids, trace_addrs = generate_prefetches(
            args.benchmark, model, config, args.twilight
        )
        inst_ids.extend(trace_ids)
        prefetch_addresses.extend(trace_addrs)
    
    # Create prefetch file
    if prefetch_addresses:
        create_prefetch_file(args.prefetch_file, inst_ids, prefetch_addresses)
        print(f"Prefetch file created: {args.prefetch_file}")
    else:
        print("No prefetches generated")

if __name__ == "__main__":
    main()