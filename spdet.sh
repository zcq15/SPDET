torchrun --rdzv_id=spdet --rdzv_backend=c10d --nproc_per_node=2\
    --rdzv_endpoint=localhost:18081 spdet.py --port 18091 --overview False\
    --id EasyPNVSDepth-SPDET --model SPDET --dataset EasyPNVSDepth\
    --imgsize 256 512 --padding circpad --supervise scatter\
    --epochs 100 --batch_size 64 --threads 32

torchrun --rdzv_id=spdet --rdzv_backend=c10d --nproc_per_node=2\
    --rdzv_endpoint=localhost:18081 spdet.py --port 18091 --overview False\
    --id HardPNVSDepth-SPDET --model SPDET --dataset HardPNVSDepth\
    --imgsize 256 512 --padding circpad --supervise scatter\
    --epochs 100 --batch_size 64 --threads 32

torchrun --rdzv_id=spdet --rdzv_backend=c10d --nproc_per_node=2\
    --rdzv_endpoint=localhost:18081 spdet.py --port 18091 --overview False\
    --id 3D60-SPDET --model SPDET --dataset 3D60\
    --imgsize 256 512 --padding circpad --supervise scatter\
    --epochs 100 --batch_size 64 --threads 32