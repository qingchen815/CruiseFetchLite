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
docker run -it --gpus all -v /path/to/data:/data tlite2 /bin/bash

2. 避免 rebuild，直接挂载修改后的文件
与其每次修改后 rebuild 镜像，可以直接挂载本地目录，像你之前那样运行容器：

bash

收起

自动换行

复制
docker run -it --gpus all -v D:\TLITE2:/app tlite2
原理：-v D:\TLITE2:/app 将本地目录挂载到容器内的 /app，修改本地文件后无需 rebuild，容器会直接使用最新版本。
适用场景：开发阶段，频繁修改代码时。
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

To train the T-LITE model on a memory trace: !

```bash
docker run -it --gpus all \
    -v $(pwd)/data:/data \
    tlite2 python train.py \
    --benchmark /data/traces/471.omnetpp-s0.txt.xz \
    --model-path /data/models/tlite2_model \
    --config /app/config/TLITE2debug1.yaml \
    --debug \
    --tb-dir /data/models/tensorboard_logs
```
single line version :
docker run -it --gpus all -v (Get-Location).Path/data:/data tlite2 python train.py --benchmark /data/traces/471.omnetpp-s0.txt.xz --model-path /data/models/tlite2_model --config /app/config/TLITE2debug1.yaml --debug --tb-dir /data/models/tensorboard_logs

### Generating Prefetches

To generate prefetch files using a trained model:

```bash
docker run -it --gpus all \
    -v $(pwd)/data:/data \
    tlite2 python generate.py \
    --model-path /data/models/tlite2_model \
    --clustering-path /data/models/clustering.npy \
    --benchmark /data/traces/471.omnetpp-s0.txt.xz \
    --output /data/models/prefetch.txt
```
single line version:
docker run -it --gpus all -v $(pwd)/data:/data tlite2 python generate.py --model-path /data/models/tlite2_model --clustering-path /data/models/clustering.npy --benchmark /data/traces/471.omnetpp-s0.txt.xz --output /data/models/prefetch.txt


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
├── app/
│   ├── config/XXX.yaml
│   ├── script/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── clustering.py
│   │   ├── example.py
│   │   ├── init.py
│   │   ├── metadata.py
│   │   ├── metrics.py
│   │   ├── model.py
│   │   └── prefetcher.py
│   ├── train.py
│   ├── generate.py
│   └── requirements.txt
├── data/
│   ├── traces/
│   └── models/
└── Dockerfile
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

mkdir -p data/traces data/models 

cp 471.omnetpp-s0.txt.xz data/traces/ 

docker run -it --gpus all \
    -v $(pwd)/data:/data \
    -p 6006:6006 \
    tlite2 tensorboard --logdir=/data/models/tensorboard_logs 