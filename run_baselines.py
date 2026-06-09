import sys, time, gc
sys.path.insert(0, "/kaggle/working/")
from shared_utils import *
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler

device = torch.device("cuda:0")
loader, _ = get_loader()
model     = build_model().to(device)
opt       = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.CrossEntropyLoss()
scaler    = GradScaler("cuda")

epoch_times, powers = [], []
for epoch in range(EPOCHS):
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0, n = time.time(), 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        with autocast("cuda"):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        n += imgs.size(0)
    dt = time.time() - t0
    epoch_times.append(dt)
    powers.append(gpu_power_watts([0]))
    print(f"  [Baseline] Epoch {epoch+1}/{EPOCHS} | {dt:.1f}s | {n/dt:.0f} samp/s", flush=True)

avg_t = sum(epoch_times) / EPOCHS
avg_p = sum(powers) / EPOCHS
save_result("baseline_1gpu", {
    "throughput_samples_per_sec": round(len(loader.dataset) / avg_t, 2),
    "avg_epoch_time_sec":         round(avg_t, 2),
    "peak_memory_mb":             round(peak_mem_mb(), 2),
    "energy_per_epoch_wh":        round(avg_p * avg_t / 3600, 4),
    "scaling_efficiency":         100.0,
    "num_gpus":                   1,
})
print(f"BASELINE_TIME={avg_t:.4f}")