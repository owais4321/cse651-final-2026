import os, sys, time
sys.path.insert(0, "/kaggle/working/")
from shared_utils import *
import torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
dist.init_process_group(backend="nccl")
rank, ws = dist.get_rank(), dist.get_world_size()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

loader, sampler = get_loader(rank, ws, distributed=True)
model = DDP(build_model().to(device), device_ids=[rank])
opt   = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
crit  = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

epoch_times, powers = [], []
for ep in range(EPOCHS):
    sampler.set_epoch(ep)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    t0, n = time.time(), 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            loss = crit(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        n += imgs.size(0)
    dt = time.time() - t0
    epoch_times.append(dt)
    if rank == 0:
        powers.append(gpu_power_watts([0,1]))
        print(f"  [DDP] Epoch {ep+1}/{EPOCHS} | {dt:.1f}s | {n*ws/dt:.0f} samp/s", flush=True)

if rank == 0:
    avg_t = sum(epoch_times)/EPOCHS
    avg_p = sum(powers)/EPOCHS
    T1    = float(os.environ.get("BASELINE_TIME", avg_t*2))
    save_result("DDP", {
        "throughput_samples_per_sec": round(len(loader.dataset)*ws/avg_t, 2),
        "avg_epoch_time_sec":         round(avg_t, 2),
        "peak_memory_mb":             round(peak_mem_mb(0), 2),
        "energy_per_epoch_wh":        round(avg_p*avg_t/3600, 4),
        "scaling_efficiency":         round(T1/(avg_t*ws)*100, 2),
        "num_gpus":                   ws,
    })
dist.destroy_process_group()