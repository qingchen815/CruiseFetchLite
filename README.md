# T-LITE2: A Lightweight Neural Prefetcher

T-LITE2 is a lightweight neural prefetcher that uses behavioral clustering and frequency-based candidate selection to predict memory accesses. It is designed to be efficient, accurate, and easy to integrate into existing systems.

## Features

- Behavioral clustering for efficient page grouping
- Frequency-based candidate selection
- Mixture-of-experts architecture for context-aware predictions
- Configurable model size and performance parameters
- Support for both training and inference
- Docker support for easy deployment

## Installation

### Using Docker (Recommended)

1. Build the Docker image:
```bash
docker build -t tlite2 .
```

2. Run the container:
```bash
docker run -it --gpus all -v /path/to/data:/data tlite2
```

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TLITE2.git
cd TLITE2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the T-LITE model on a memory trace:

```bash
python train.py \
    --benchmark /path/to/trace.txt \
    --model-path /path/to/save/model \
    --config configs/baseTLITE2.yaml \
    --tb-dir /path/to/tensorboard/logs
```

### Generating Prefetches

To generate prefetch files using a trained model:

```bash
python generate.py \
    --model-path /path/to/model \
    --clustering-path /path/to/clustering.npy \
    --benchmark /path/to/trace.txt \
    --output /path/to/prefetch.txt \
    --config configs/baseTLITE2.yaml
```

### Example Usage

Run the example script to test the model with synthetic data:

```bash
python example.py --simulate
```

## Configuration

The model can be configured through a YAML file. See `configs/baseTLITE2.yaml` for all available parameters:

### Key Parameters

- `num_clusters`: Number of behavioral clusters (default: 4096)
- `history_length`: Number of history entries to consider (default: 3)
- `num_candidates`: Number of candidate pages for prefetching (default: 4)
- `learning_rate`: Initial learning rate (default: 0.001)
- `batch_size`: Training batch size (default: 256)
- `epochs`: Maximum number of training epochs (default: 500)

### Memory Management

- `max_pages`: Maximum number of pages to track
- `cache_size_kb`: Size of metadata cache
- `cache_ways`: Number of ways in metadata cache

### Performance Tuning

- `enable_profiling`: Enable detailed performance profiling
- `profile_interval`: Number of accesses between profile snapshots
- `num_workers`: Number of worker threads for data processing

## Project Structure

```
TLITE2/
├── configs/
│   └── baseTLITE2.yaml    # Base configuration file
├── TLITE2file/
│   ├── clustering.py      # Behavioral clustering utilities
│   ├── config.py         # Configuration management
│   ├── example.py        # Example usage
│   ├── generate.py       # Prefetch generation
│   ├── init.py          # Package initialization
│   ├── metadata.py      # Metadata management
│   ├── metrics.py       # Performance metrics
│   ├── model.py         # T-LITE model implementation
│   ├── prefetcher.py    # Prefetcher implementation
│   └── train.py         # Training script
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Performance Metrics

The model tracks several performance metrics:

- Accuracy: Percentage of useful prefetches
- Coverage: Percentage of memory accesses covered by prefetches
- Prediction Latency: Average time to generate predictions
- Metadata Size: Size of clustering and metadata information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use T-LITE2 in your research, please cite:

```bibtex
@software{tlite2,
  title = {T-LITE2: A Lightweight Neural Prefetcher},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/TLITE2}
}
``` 