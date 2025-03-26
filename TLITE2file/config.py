import yaml

class ModelConfig:
    '''Configuration class for T-LITE model parameters'''
    
    def __init__(self):
        # Model structure parameters
        self.pc_embed_size = 64
        self.cluster_embed_size = 25
        self.offset_embed_size = 2500  # cluster_embed_size * num_experts
        self.num_experts = 100
        self.history_length = 3
        self.num_pcs = 4096
        self.num_clusters = 4096
        self.offset_size = 64
        self.num_candidates = 4
        self.dpf_history_length = 1
        self.offset_bits = 6
        
        # Training parameters
        self.learning_rate = 0.001
        self.lr_decay_rate = 0.5
        self.batch_size = 256
        self.epochs = 500
        self.early_stopping_patience = 50
        
        # Additional parameters
        self.steps_per_epoch = 80000
        self.quantize_bits = 8
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """Load configuration from a YAML file"""
        config = cls()
        
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Update config with values from YAML
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
        
        return config
    
    def to_yaml(self, yaml_path):
        """Save configuration to a YAML file"""
        # Get all attributes
        config_dict = {k: v for k, v in self.__dict__.items()}
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __str__(self):
        """String representation of configuration"""
        config_str = "ModelConfig:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str


def extend_voyager_config_for_tlite(yaml_path, output_path=None):
    """
    Extend a Voyager configuration YAML file with T-LITE specific parameters
    
    Args:
        yaml_path: Path to Voyager configuration YAML
        output_path: Path to save extended configuration, if None uses yaml_path
    
    Returns:
        Extended configuration object
    """
    # Load Voyager config
    with open(yaml_path, 'r') as f:
        voyager_config = yaml.safe_load(f)
    
    # Add T-LITE specific parameters
    tlite_params = {
        'num_candidates': 4,
        'dpf_history_length': 1,
        'num_clusters': 4096,
        'quantize_bits': 8
    }
    
    # Merge configurations
    for key, value in tlite_params.items():
        if key not in voyager_config:
            voyager_config[key] = value
    
    # Save extended configuration
    if output_path is not None:
        with open(output_path, 'w') as f:
            yaml.dump(voyager_config, f, default_flow_style=False)
    elif output_path != yaml_path:
        with open(yaml_path, 'w') as f:
            yaml.dump(voyager_config, f, default_flow_style=False)
    
    # Create config object
    config = ModelConfig()
    for key, value in voyager_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
