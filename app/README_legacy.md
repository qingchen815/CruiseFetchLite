# CruiseFetchLite: A Lightweight Neural Prefetcher

CruiseFetchLite is a lightweight neural prefetcher that uses behavioral clustering and frequency-based candidate selection to predict memory accesses. It is designed to be efficient, accurate, and easy to integrate into existing systems.

## Features

- Behavioral clustering for efficient page grouping
- Frequency-based candidate selection
- Multi-stream prefetching with dynamic load balancing
- Context-aware predictions using attention mechanism
- Configurable model size and performance parameters
- Support for both static and dynamic page-to-cluster mapping
- Docker support for easy deployment

## Installation

### Using Docker (Recommended)

1. Build the Docker image:
```bash
docker build -t tlite2 .
```

2. Run the container with local directory mounting (recommended for development):
```bash
docker run -it --gpus all -v /path/to/local/dir:/app tlite2
```

This mounting approach (-v) allows you to modify code without rebuilding the container.

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

### Generating Prefetches

To generate prefetch files using a trained model:

```bash
python generate.py \
    --model-path /data/models/tlite2_model \
    --clustering-path /data/models/clustering.npy \
    --benchmark /data/traces/471.omnetpp-s0.txt.xz \
    --output /data/models/prefetch.txt \
    --config ./config/TLITE2debug1.yaml
```

Or using Docker:
```bash
docker run -it --gpus all -v $(pwd)/data:/data tlite2 python generate.py \
    --model-path /data/models/tlite2_model \
    --clustering-path /data/models/clustering.npy \
    --benchmark /data/traces/471.omnetpp-s0.txt.xz \
    --output /data/models/prefetch.txt
```

## Configuration

The model can be configured through a YAML file. Key parameters include:

### Model Parameters
- `num_clusters`: Number of behavioral clusters
- `history_length`: Number of history entries to consider
- `num_candidates`: Number of candidate pages for prefetching
- `offset_bits`: Number of bits for offset calculation

### Stream Management
- `num_streams`: Number of parallel prefetching streams (default: 32)
- `batch_size`: Processing batch size (default: 50000)

### Performance Optimization
- `enable_jit`: Enable TensorFlow JIT compilation
- `enable_debug`: Enable detailed debug outputs
- `stream_rebalance_interval`: Interval for checking stream load balance

## Project Structure

```
CruiseFetchLite/
├── app/
│ ├── config/
│ │ └── TLITE2debug1.yaml
│ ├── script/
│ │ ├── model.py # Neural network model implementation
│ │ ├── prefetcher.py # Prefetcher implementation
│ │ ├── metadata.py # Metadata management
│ │ └── config.py # Configuration handling
│ └── generate.py # Main prefetch generation script
├── data/
│ ├── traces/ # Memory access traces
│ └── models/ # Trained models and outputs
└── Dockerfile
```

## Performance Metrics

The model tracks several performance metrics:

- Accuracy: Percentage of useful prefetches
- Coverage: Percentage of memory accesses covered by prefetches
- Prediction Latency: Average time to generate predictions
- Metadata Size: Size of clustering and metadata information

## Performance Features

### Multi-Stream Processing
- Parallel processing using multiple independent prefetching streams
- Dynamic load balancing across streams
- Stream allocation based on PC and address hashing

### Clustering Management
- Static clustering from pre-trained model
- Dynamic clustering for new pages
- Efficient page-to-cluster mapping

### Optimization Techniques
- Batch processing for GPU efficiency
- Pre-allocated arrays for reduced memory allocation
- JIT compilation for improved performance
- Vectorized operations for prediction processing

## Monitoring and Debug Features

The system provides detailed monitoring capabilities:
- Stream load distribution statistics
- Prefetch generation rates
- Clustering mapping effectiveness
- Memory usage and batch processing statistics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Setup Data Directory

```bash
mkdir -p data/traces data/models
cp your_trace_file.txt.xz data/traces/
```

## Note

This is an active development project. For the latest features and updates, please check the repository regularly.

## Citation

If you use T-LITE2 in your research, please cite:

```bibtex
@software{CruiseFetchLite,
  title = {CruiseFetchLite: A Lightweight Neural Prefetcher},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/KevinMedicine26/CruiseFetchLite}
}
```

mkdir -p data/traces data/models 

cp 471.omnetpp-s0.txt.xz data/traces/ 

docker run -it --gpus all \
    -v $(pwd)/data:/data \
    -p 6006:6006 \
    tlite2 tensorboard --logdir=/data/models/tensorboard_logs 