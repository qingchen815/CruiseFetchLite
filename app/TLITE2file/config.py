import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class TLITEConfig:
    """Configuration for T-LITE neural prefetcher"""
    
    # Model Structure Parameters
    pc_embed_size: int = 64
    cluster_embed_size: int = 25
    offset_embed_size: int = 2500  # cluster_embed_size * num_experts
    num_experts: int = 100
    history_length: int = 3
    num_pcs: int = 4096
    num_clusters: int = 4096
    offset_size: int = 64
    num_candidates: int = 4
    dpf_history_length: int = 1
    offset_bits: int = 6
    
    # Training Parameters
    learning_rate: float = 0.001
    lr_decay_rate: float = 0.5
    batch_size: int = 256
    epochs: int = 500
    early_stopping_patience: int = 50
    steps_per_epoch: int = 80000
    
    # Quantization Parameters
    quantize_bits: int = 8
    
    # Prefetcher Parameters
    use_deltas: bool = True
    multi_label: bool = False
    global_stream: bool = False
    pc_localized: bool = True
    global_output: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TLITEConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()}

def load_config(yaml_path: str) -> TLITEConfig:
    """Load configuration from YAML file"""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return TLITEConfig.from_dict(config_dict)

def save_config(config: TLITEConfig, yaml_path: str):
    """Save configuration to YAML file"""
    with open(yaml_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

def extend_voyager_config_for_tlite(voyager_config: dict) -> dict:
    """
    Extend Voyager configuration with TLITE-specific parameters
    
    Args:
        voyager_config: Dictionary containing Voyager configuration
        
    Returns:
        Extended configuration dictionary with TLITE parameters
    """
    # TLITE-specific parameters with default values
    tlite_params = {
        'pc_embed_size': 64,
        'cluster_embed_size': 25,
        'offset_embed_size': 2500,
        'num_experts': 100,
        'history_length': 3,
        'num_pcs': 4096,
        'num_clusters': 4096,
        'offset_size': 64,
        'num_candidates': 4,
        'dpf_history_length': 1,
        'offset_bits': 6,
        'learning_rate': 0.001,
        'lr_decay_rate': 0.5,
        'batch_size': 256,
        'epochs': 500,
        'early_stopping_patience': 50,
        'steps_per_epoch': 80000,
        'quantize_bits': 8,
        'use_deltas': True,
        'multi_label': False,
        'global_stream': False,
        'pc_localized': True,
        'global_output': False
    }
    
    # Merge configurations, preserving Voyager values if present
    for key, value in tlite_params.items():
        if key not in voyager_config:
            voyager_config[key] = value
    
    return voyager_config

def parse_args():
    """Parse command line arguments for TLITE Neural Prefetcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TLITE Neural Prefetcher')
    
    # Training arguments
    parser.add_argument('--benchmark', type=str, required=True,
                      help='Path to benchmark trace file')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to save/load model')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with smaller dataset')
    parser.add_argument('--tb-dir', type=str,
                      help='TensorBoard log directory')
    
    # Generation arguments
    parser.add_argument('--prefetch-file', type=str,
                      help='Path to output prefetch file')
    parser.add_argument('--train', action='store_true',
                      help='Generate for training dataset')
    parser.add_argument('--valid', action='store_true',
                      help='Generate for validation dataset')
    parser.add_argument('--no-test', action='store_true',
                      help='Skip test dataset')
    
    return parser.parse_args()
