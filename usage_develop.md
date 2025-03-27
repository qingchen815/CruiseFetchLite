build

docker build -t tlite2 .

start

docker run -it --gpus all -v "D:\TLITE2\app:/app" -v "D:\TLITE2\data:/data" tlite2 /bin/bash



train
python train.py \
    --benchmark /data/traces/471.omnetpp-s0.txt.xz \
    --model-path /data/models/tlite2_model \
    --config /app/config/TLITE2debug1.yaml \
    --debug \
    --tb-dir /data/models/tensorboard_logs





python train.py --benchmark /data/traces/471.omnetpp-s0.txt.xz --model-path /data/models/tlite2_model --config /app/config/TLITE2debug1.yaml --debug --tb-dir /data/models/tensorboard_logs

generate command

python generate.py --model-path /data/models/tlite2_model --clustering-path /data/models/clustering.npy --benchmark /data/traces/471.omnetpp-s0.txt.xz --output /data/models/prefetch3.txt

