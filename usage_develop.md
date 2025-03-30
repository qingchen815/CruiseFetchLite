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

python generate.py \--model-path /data/models/CFlitebase1_model \--clustering-path /data/models/clustering.npy \--benchmark /data/traces/471.omnetpp-s0.txt.xz \--output /data/models/prefetch_base1_471tpp0.txt \--config /app/config/base1.yaml



advance debug generate new

python generate.py --model-path /path/to/model --clustering-path /path/to/clustering.npy --benchmark /path/to/trace.txt --output /path/to/output.txt --sequential --debug-file debug.log --prefetch-distance 100
参数说明:

--sequential: 启用顺序处理逻辑（强烈推荐）
--test-clustering: 运行聚类映射诊断
--debug-file: 输出详细调试信息
--prefetch-distance: 设置预取距离

这个版本保留了高效处理的优势，同时确保了正确的访问顺序和元数据更新连贯性。希望这能解决您的预取性能问题！

new generate
python generate.py --model-path /data/models/CFlitebase1_model --clustering-path /data/models/clustering.npy --benchmark /data/traces/471.omnetpp-s0.txt.xz --output /data/models/prefetch_base1_471tpp0.txt --sequential --debug-file /data/models/debug_471tpp0.log --prefetch-distance 100 --config ./config/base1.yaml



'--config', default='./config/base1.yaml'