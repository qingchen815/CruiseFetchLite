import numpy as np
import tensorflow as tf
import os
import argparse

from config import ModelConfig
from model import create_tlite_model
from prefetcher import TLITEPrefetcher
from clustering import BehavioralClusteringUtils

def parse_args():
    parser = argparse.ArgumentParser(description='T-LITE Example')
    parser.add_argument('--model-path', type=str, default=None, help='Path to load a trained model')
    parser.add_argument('--simulate', action='store_true', help='Run a simulation with synthetic memory accesses')
    parser.add_argument('--quantize', action='store_true', help='Quantize the model to 8-bit precision')
    return parser.parse_args()

def create_sample_model():
    """Create a sample T-LITE model for demonstration"""
    print("Creating sample T-LITE model...")
    config = ModelConfig()
    
    # Reduce model size for demonstration
    config.num_clusters = 256
    config.num_pcs = 256
    config.history_length = 2
    config.num_candidates = 4
    
    model = create_tlite_model(config)
    print(f"Model created with {sum(np.prod(v.get_shape()) for v in model.trainable_variables):,} parameters")
    
    return model, config

def simulate_memory_accesses(model, config, num_accesses=1000):
    """Simulate memory accesses and test the prefetcher"""
    print(f"\nSimulating {num_accesses} memory accesses...")
    
    # Create prefetcher
    prefetcher = TLITEPrefetcher(model, config)
    
    # Generate synthetic memory traces
    
    # Create a few patterns to simulate temporal locality
    patterns = [
        # Pattern 1: Sequential access with stride 1
        lambda i: 0x1000 + (i % 20) * 64,
        
        # Pattern 2: Sequential access with stride 2
        lambda i: 0x2000 + ((i % 10) * 2) * 64,
        
        # Pattern 3: Random access among 5 addresses
        lambda i: 0x3000 + np.random.choice([0, 1, 2, 3, 4]) * 64,
        
        # Pattern 4: Linked list traversal (simulated)
        lambda i: 0x4000 + ([0, 3, 1, 4, 2] * 10)[i % 50] * 64,
    ]
    
    # PC values for each pattern
    pattern_pcs = {
        0: 0x100,  # PC for pattern 1
        1: 0x200,  # PC for pattern 2
        2: 0x300,  # PC for pattern 3
        3: 0x400,  # PC for pattern 4
    }
    
    # Create a schedule mixing the patterns
    schedule = []
    for i in range(num_accesses):
        # Choose pattern based on position
        pattern_idx = (i // 20) % len(patterns)
        addr = patterns[pattern_idx](i)
        pc = pattern_pcs[pattern_idx]
        schedule.append((addr, pc))
    
    # Run simulation
    prefetches = 0
    correct_prefetches = 0
    
    for i, (addr, pc) in enumerate(schedule):
        # Get prefetch for current access
        prefetch = prefetcher.handle_access(addr, pc)
        
        if prefetch is not None:
            prefetches += 1
            
            # Check if prefetch is useful (appears in next 10 accesses)
            future_window = schedule[i+1:i+11] if i < len(schedule) - 11 else schedule[i+1:]
            future_addrs = [a for a, _ in future_window]
            
            if prefetch in future_addrs:
                correct_prefetches += 1
    
    # Print results
    print("\nSimulation Results:")
    print(f"  Total accesses: {num_accesses}")
    print(f"  Prefetches issued: {prefetches}")
    print(f"  Correct prefetches: {correct_prefetches}")
    if prefetches > 0:
        print(f"  Accuracy: {correct_prefetches / prefetches * 100:.2f}%")
    print(f"  Coverage: {correct_prefetches / num_accesses * 100:.2f}%")
    
    # Print prefetcher internal stats
    prefetcher.print_stats()

def quantize_model(model):
    """Quantize model weights and show size reduction"""
    print("\nQuantizing model...")
    
    # Get size before quantization
    weights = model.get_weights()
    orig_size_bytes = sum(w.nbytes for w in weights)
    
    # Apply quantization
    stats = model.quantize(bits=8)
    
    print(f"Original size: {orig_size_bytes / (1024 * 1024):.2f} MB")
    print(f"Quantized size ({stats['bits']}-bit): {stats['quant_size_mb']:.2f} MB")
    print(f"Size reduction: {(1 - stats['quant_size_mb'] / (orig_size_bytes / (1024 * 1024))) * 100:.2f}%")
    
    return model

def profile_inference(model, config):
    """Profile model inference latency"""
    print("\nProfiling inference latency...")
    
    # Create dummy inputs
    batch_size = 1
    history_length = config.history_length
    num_candidates = config.num_candidates
    dpf_history_length = config.dpf_history_length
    
    dummy_cluster_history = np.zeros((batch_size, history_length), dtype=np.int32)
    dummy_offset_history = np.zeros((batch_size, history_length), dtype=np.int32)
    dummy_pc = np.zeros((batch_size, 1), dtype=np.int32)
    dummy_dpf = np.zeros((batch_size, dpf_history_length, num_candidates), dtype=np.float32)
    
    # Warm up
    for _ in range(10):
        model((dummy_cluster_history, dummy_offset_history, dummy_pc, dummy_dpf))
    
    # Time inference
    import time
    num_inferences = 1000
    
    start_time = time.time()
    for _ in range(num_inferences):
        model((dummy_cluster_history, dummy_offset_history, dummy_pc, dummy_dpf))
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_inferences * 1000
    inferences_per_second = num_inferences / (end_time - start_time)
    
    print(f"Average inference time: {avg_time_ms:.3f} ms")
    print(f"Inferences per second: {inferences_per_second:.1f}")
    
    # Estimate FLOPs
    # This is a rough calculation based on the model architecture
    # For a more accurate count, use a profiling tool
    num_params = sum(np.prod(v.get_shape()) for v in model.trainable_variables)
    estimated_flops = num_params * 2  # Assume 2 FLOPs per parameter on average
    
    print(f"Estimated FLOPs per inference: {estimated_flops:,}")

def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    if args.model_path and os.path.exists(args.model_path):
        # Load existing model
        print(f"Loading model from {args.model_path}...")
        # This is a placeholder - the actual loading would depend on your model format
        # model = tf.keras.models.load_model(args.model_path)
        raise NotImplementedError("Model loading not implemented in this example")
    else:
        # Create a sample model
        model, config = create_sample_model()
    
    # Quantize if requested
    if args.quantize:
        model = quantize_model(model)
    
    # Profile inference
    profile_inference(model, config)
    
    # Run simulation if requested
    if args.simulate:
        simulate_memory_accesses(model, config)
    
    print("\nExample completed.")

if __name__ == "__main__":
    main()
