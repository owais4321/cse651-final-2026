import os, time, json, subprocess
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
from torchvision.models import resnet50

# ── Hyperparameters ────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 5
LR          = 0.01
NUM_WORKERS = 2
DATA_DIR    = "/tmp/cifar10"
RESULTS_PATH = "/tmp/benchmark_results.json"

def get_loader(rank=0, world_size=1, distributed=False):
    tfm = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=tfm)
    sampler = (
        torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True)
    return loader, sampler

def build_model():
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m

def peak_mem_mb(device_id=0):
    return torch.cuda.max_memory_allocated(device_id) / 1024**2

def gpu_power_watts(device_ids):
    ids = ",".join(str(i) for i in device_ids)
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={ids}",
         "--query-gpu=power.draw",
         "--format=csv,noheader,nounits"], text=True
    ).strip().split("\\n")
    return sum(float(w) for w in out if w.strip())

def save_result(framework, data):
    try:
        with open(RESULTS_PATH) as f: all_r = json.load(f)
    except FileNotFoundError:
        all_r = {}
    all_r[framework] = data
    with open(RESULTS_PATH, "w") as f: json.dump(all_r, f, indent=2)
    print(f"[{framework}] saved → {RESULTS_PATH}")