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
