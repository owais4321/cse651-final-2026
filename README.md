# Distributed Deep Learning Benchmark: DDP vs Horovod vs FSDP-Zero1

A benchmarking study comparing three distributed training strategies — PyTorch DDP, Horovod-style ring-allreduce, and FSDP-Zero1 — on dual-GPU commodity hardware (Kaggle T4 x2).

---

## What This Benchmarks

| Strategy | Gradient Sync | Memory Saving | Backend |
|---|---|---|---|
| PyTorch DDP | Bucketed AllReduce | None | NCCL |
| Horovod-style | Ring-AllReduce | None | NCCL |
| FSDP-Zero1 | ReduceScatter + AllGather | Optimizer states sharded | NCCL |

**Model:** ResNet-50  
**Dataset:** Synthetic ImageNet-style (224×224, 1000 classes)  
**Precision:** FP16 via `torch.amp` (mixed precision)  
**Batch size:** 64 per GPU

---

## Metrics Reported

- Throughput (samples/sec)
- Scaling efficiency vs single-GPU baseline
- Peak GPU memory usage (MB)
- Energy per epoch (Wh)
- Average epoch time (s)

---

## Project Structure

```
├── distributed_dl_benchmark_v3.ipynb   # Main notebook
├── run_baseline.py                     # Single-GPU baseline (subprocess)
├── run_ddp.py                          # DDP training script
├── run_horovod.py                      # Horovod-style training script
├── run_fsdp.py                         # FSDP-Zero1 training script
├── shared_utils.py                     # Shared config, loader, model builder
└── README.md
```

---

## How to Run

### On Kaggle (recommended)
1. Upload the notebook to Kaggle
2. Enable **GPU T4 x2** accelerator under Settings
3. Run all cells top to bottom — each strategy launches via `torchrun` as a subprocess

### Locally (2× CUDA GPUs required)
```bash
pip install torch torchvision

# Single GPU baseline
python run_baseline.py

# DDP
torchrun --nproc_per_node=2 --master_port=29500 run_ddp.py

# Horovod-style
torchrun --nproc_per_node=2 --master_port=29501 run_horovod.py

# FSDP-Zero1
torchrun --nproc_per_node=2 --master_port=29502 run_fsdp.py
```

---

## Key Implementation Notes

**Each strategy runs as a separate subprocess** — this is intentional. The Kaggle notebook kernel holds GPU memory after the baseline run; launching `torchrun` workers as subprocesses ensures they start with a clean GPU memory state.

**Mixed precision is enabled on all runs** via `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')` to keep peak memory within T4 limits (15 GB per card).

**`PYTORCH_ALLOC_CONF=expandable_segments:True`** is set in each training script to reduce memory fragmentation.

**DataLoader workers** are set to 2 per GPU (not 4) to avoid process contention under `torchrun`.

---

## Requirements

```
torch >= 2.1
torchvision >= 0.16
python >= 3.10
CUDA >= 11.8
2× GPU (tested on Tesla T4 16GB)
```

---

## Results Summary

Results are saved to `benchmark_results.json` after each run and printed as a comparison table at the end of the notebook.
