CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_id=spdet --rdzv_backend=c10d --nproc_per_node=2\
    --rdzv_endpoint=localhost:18081 spdet.py --port 18091\
    --id Un-3D60-SPDET --model SPDET --dataset 3D60\
    --supervise scatter --optim Adam --lr 1e-4\
    --epochs 100 --batch_size 64 --threads 16