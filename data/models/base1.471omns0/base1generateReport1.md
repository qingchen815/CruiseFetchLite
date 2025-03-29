build

docker build -t tlite2 .

start

docker run -it --gpus all -v "D:\TLITE2\app:/app" -v "D:\TLITE2\data:/data" tlite2 /bin/bash

train
python train.py \
 --benchmark /data/traces/471.omnetpp-s0.txt.xz \
 --model-path /data/models/CFlite_model_base1 \
 --config /app/config/base1.yaml \
 --debug \
 --tb-dir /data/models/tensorboard_logs_base1

TLITE2debug1.yaml

python train.py --benchmark /data/traces/471.omnetpp-s0.txt.xz --model-path /data/models/CFlitebase1_model --config /app/config/base1.yaml --debug --tb-dir /data/models/tensorboard_logs_base1

generate command

python generate.py --model-path /data/models/CFlitebase1_model --clustering-path /data/models/clustering.npy --benchmark /data/traces/471.omnetpp-s0.txt.xz --output /data/models/prefetchbase1.txt

python generate.py \
 --model-path /data/models/CFlitebase1_model \
 --clustering-path /data/models/clustering.npy \
 --benchmark /data/traces/471.omnetpp-s0.txt.xz \
 --output /data/models/prefetchbase1.txt \
 --config /app/config/base1.yaml

python generate.py \--model-path /data/models/CFlitebase1_model \--clustering-path /data/models/clustering.npy \--benchmark /data/traces/471.omnetpp-s0.txt.xz \--output /data/models/prefetchbase1.txt \--config /app/config/base1.yaml

PS D:\TLITE2> docker build -t tlite2 .
ERROR: error during connect: Head "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/\_ping": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
PS D:\TLITE2> docker run -it --gpus all -v "D:\TLITE2\app:/app" -v "D:\TLITE2\data:/data" tlite2 /bin/bash

==========
== CUDA ==
==========

CUDA Version 11.8.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

root@d4949a3a8cf1:/app# python generate.py \--model-path /data/models/CFlitebase1_model \--clustering-path /data/models/clustering.npy \--benchmark /data/traces/471.omnetpp-s0.txt.xz \--output /data/models/prefetchbase1.txt \--config /app/config/base1.yaml
2025-03-29 05:31:45.985373: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-03-29 05:31:47.622553: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT

=== Loading Model ===
2025-03-29 05:31:51.845930: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
2025-03-29 05:31:52.229917: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
2025-03-29 05:31:52.230366: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
2025-03-29 05:31:52.233396: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
2025-03-29 05:31:52.233828: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
2025-03-29 05:31:52.234317: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
2025-03-29 05:31:53.195202: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
2025-03-29 05:31:53.195556: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
2025-03-29 05:31:53.195593: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1722] Could not identify NUMA node of platform GPU id 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2025-03-29 05:31:53.195962: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
2025-03-29 05:31:53.196023: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3586 MB memory: -> device: 0, name: NVIDIA GeForce RTX 3060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6
2025-03-29 05:31:54.477339: I tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:637] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
Warning: Could not import metrics module, using default metrics
Model weights loaded successfully. Model structure:
Model: "tlite"

---

# Layer (type) Output Shape Param

embedding (Embedding) multiple 262144

embedding_1 (Embedding) multiple 102400

embedding_2 (Embedding) multiple 160000

multi_head_attention (Multi multiple 2600
HeadAttention)

dense (Dense) multiple 1095

dense_1 (Dense) multiple 14016

=================================================================
Total params: 542,255
Trainable params: 542,255
Non-trainable params: 0

---

None

=== Loading Clustering Information ===
Clustering information loaded. Available keys: dict_keys(['page_to_cluster', 'cluster_offset_transitions', 'cluster_successors'])
Clustering information contains 20343 page mappings
Random sample of 5 mappings:
Page 94d992fc -> Cluster 3337
Page a98092379 -> Cluster 2408
Page ac6d9baba -> Cluster 2894
Page faef35a0a -> Cluster 2424
Page 49950aa9f -> Cluster 2745

=== Starting Prefetch Generation ===
Processing trace file: /data/traces/471.omnetpp-s0.txt.xz
Output file: /data/models/prefetchbase1.txt

=== Clustering Information Analysis ===
Page ID Range: b381a39 - ff3c8d5b9
Number of Clusters: 4096
Number of Mapped Pages: 20343
Using 32 parallel streams
Batch Processing Size: 20000
Calculating total lines...
Total lines to process: 8940571
2025-03-29 05:32:06.939818: I tensorflow/compiler/xla/service/service.cc:169] XLA service 0x2dbd95e0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2025-03-29 05:32:06.939870: I tensorflow/compiler/xla/service/service.cc:177] StreamExecutor device (0): NVIDIA GeForce RTX 3060 Laptop GPU, Compute Capability 8.6
2025-03-29 05:32:07.017084: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2025-03-29 05:32:07.217521: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8906
2025-03-29 05:32:09.231726: I ./tensorflow/compiler/jit/device_compiler.h:180] Compiled cluster using XLA! This line is logged at most once for the lifetime of the process.
Progress: 200000/8940571 (2.24%)
Current Prefetch Rate: 32.40%
Progress: 400000/8940571 (4.47%)
Current Prefetch Rate: 31.67%
Progress: 600000/8940571 (6.71%)
Current Prefetch Rate: 32.08%
Progress: 800000/8940571 (8.95%)
Current Prefetch Rate: 32.62%
Progress: 1000000/8940571 (11.18%)
Current Prefetch Rate: 33.28%
Progress: 1200000/8940571 (13.42%)
Current Prefetch Rate: 33.49%
Progress: 1400000/8940571 (15.66%)
Current Prefetch Rate: 33.44%
Progress: 1600000/8940571 (17.90%)
Current Prefetch Rate: 32.75%
Progress: 1800000/8940571 (20.13%)
Current Prefetch Rate: 32.73%
Progress: 2000000/8940571 (22.37%)
Current Prefetch Rate: 32.69%

=== Stream Load Distribution ===
Stream 19: 3.34% (66869 updates)
Stream 8: 3.30% (66068 updates)
Stream 11: 3.28% (65644 updates)
Stream 21: 3.27% (65387 updates)
Stream 24: 3.23% (64556 updates)

Active Streams: 32/32
Average Load: 62500 updates/stream
Load Range: 58489 - 66869 updates
Max/Min Load Ratio: 1.14x
Progress: 2200000/8940571 (24.61%)
Current Prefetch Rate: 32.42%
Progress: 2400000/8940571 (26.84%)
Current Prefetch Rate: 32.03%
Progress: 2600000/8940571 (29.08%)
Current Prefetch Rate: 32.00%
Progress: 2800000/8940571 (31.32%)
Current Prefetch Rate: 31.93%
Progress: 3000000/8940571 (33.55%)
Current Prefetch Rate: 31.93%
Progress: 3200000/8940571 (35.79%)
Current Prefetch Rate: 31.75%
Progress: 3400000/8940571 (38.03%)
Current Prefetch Rate: 31.79%
Progress: 3600000/8940571 (40.27%)
Current Prefetch Rate: 31.67%
Progress: 3800000/8940571 (42.50%)
Current Prefetch Rate: 31.39%
Progress: 4000000/8940571 (44.74%)
Current Prefetch Rate: 31.31%

=== Stream Load Distribution ===
Stream 19: 3.35% (134017 updates)
Stream 8: 3.32% (132688 updates)
Stream 11: 3.27% (130757 updates)
Stream 21: 3.27% (130713 updates)
Stream 24: 3.23% (129374 updates)

Active Streams: 32/32
Average Load: 125000 updates/stream
Load Range: 117019 - 134017 updates
Max/Min Load Ratio: 1.15x
Progress: 4200000/8940571 (46.98%)
Current Prefetch Rate: 31.45%
Progress: 4400000/8940571 (49.21%)
Current Prefetch Rate: 31.43%
Progress: 4600000/8940571 (51.45%)
Current Prefetch Rate: 31.47%
Progress: 4800000/8940571 (53.69%)
Current Prefetch Rate: 31.66%
Progress: 5000000/8940571 (55.92%)
Current Prefetch Rate: 31.75%
Progress: 5200000/8940571 (58.16%)
Current Prefetch Rate: 31.88%
Progress: 5400000/8940571 (60.40%)
Current Prefetch Rate: 31.90%
Progress: 5600000/8940571 (62.64%)
Current Prefetch Rate: 31.87%
Progress: 5800000/8940571 (64.87%)
Current Prefetch Rate: 31.88%
Progress: 6000000/8940571 (67.11%)
Current Prefetch Rate: 31.82%

=== Stream Load Distribution ===
Stream 19: 3.36% (201387 updates)
Stream 8: 3.33% (199622 updates)
Stream 21: 3.27% (195966 updates)
Stream 11: 3.26% (195880 updates)
Stream 24: 3.25% (194971 updates)

Active Streams: 32/32
Average Load: 187500 updates/stream
Load Range: 175786 - 201387 updates
Max/Min Load Ratio: 1.15x
Progress: 6200000/8940571 (69.35%)
Current Prefetch Rate: 31.70%
Progress: 6400000/8940571 (71.58%)
Current Prefetch Rate: 31.61%
Progress: 6600000/8940571 (73.82%)
Current Prefetch Rate: 31.57%
Progress: 6800000/8940571 (76.06%)
Current Prefetch Rate: 31.42%
Progress: 7000000/8940571 (78.29%)
Current Prefetch Rate: 31.38%
Progress: 7200000/8940571 (80.53%)
Current Prefetch Rate: 31.40%
Progress: 7400000/8940571 (82.77%)
Current Prefetch Rate: 31.44%
Progress: 7600000/8940571 (85.01%)
Current Prefetch Rate: 31.30%
Progress: 7800000/8940571 (87.24%)
Current Prefetch Rate: 31.33%
Progress: 8000000/8940571 (89.48%)
Current Prefetch Rate: 31.29%

=== Stream Load Distribution ===
Stream 19: 3.36% (268732 updates)
Stream 8: 3.33% (266457 updates)
Stream 11: 3.26% (261098 updates)
Stream 21: 3.26% (260970 updates)
Stream 24: 3.26% (260841 updates)

Active Streams: 32/32
Average Load: 250000 updates/stream
Load Range: 234639 - 268732 updates
Max/Min Load Ratio: 1.15x
Progress: 8200000/8940571 (91.72%)
Current Prefetch Rate: 31.28%
Progress: 8400000/8940571 (93.95%)
Current Prefetch Rate: 31.28%
Progress: 8600000/8940571 (96.19%)
Current Prefetch Rate: 31.24%
Progress: 8800000/8940571 (98.43%)
Current Prefetch Rate: 31.27%

=== Final Statistics ===
Total Lines Processed: 8940000
Total Predictions: 8940000
Valid Prefetches: 2794338
Overall Prefetch Rate: 31.26%

=== Final Stream Load Distribution ===

Top 10 Most Active Streams:
Stream 19:
Load: 3.36% (300350 updates)
Prefetches: 30.94% (92941 prefetches)
Stream 8:
Load: 3.33% (297610 updates)
Prefetches: 35.17% (104659 prefetches)
Stream 24:
Load: 3.26% (291795 updates)
Prefetches: 38.73% (113016 prefetches)
Stream 11:
Load: 3.26% (291699 updates)
Prefetches: 31.15% (90878 prefetches)
Stream 21:
Load: 3.26% (291537 updates)
Prefetches: 34.32% (100065 prefetches)
Stream 29:
Load: 3.22% (287624 updates)
Prefetches: 28.97% (83315 prefetches)
Stream 12:
Load: 3.20% (286311 updates)
Prefetches: 29.92% (85677 prefetches)
Stream 13:
Load: 3.19% (285242 updates)
Prefetches: 24.61% (70198 prefetches)
Stream 0:
Load: 3.18% (284425 updates)
Prefetches: 37.15% (105666 prefetches)
Stream 23:
Load: 3.17% (283094 updates)
Prefetches: 33.04% (93530 prefetches)

Load Distribution Statistics:
Active Streams: 32/32
Average Load: 279393 updates/stream
Load Range: 262625 - 300350 updates
Max/Min Load Ratio: 1.14x
Prefetch generation complete!
root@d4949a3a8cf1:/app#
