import time
import numpy as np
import torch
import torch.nn as nn
from aaronnet import Tensor, Sequential, Linear, ReLU

def benchmark_mlp(size=512, hidden=1024, batch_size=32, iterations=80):
    # ─── AaronNet ────────────────────────────────────────────────────────────
    model = Sequential(
        Linear(size, hidden),
        ReLU(),
        Linear(hidden, size)
    )

    x_np = Tensor(np.random.randn(batch_size, size))

    # Warmup
    _ = model(x_np).sum().backward()

    start = time.perf_counter()
    for _ in range(iterations):
        for p in model.parameters():
            p.grad = None
        out = model(x_np)
        out.sum().backward()
    aaron_time = (time.perf_counter() - start) / iterations

    # ─── PyTorch ─────────────────────────────────────────────────────────────
    model_t = nn.Sequential(
        nn.Linear(size, hidden),
        nn.ReLU(),
        nn.Linear(hidden, size)
    ).to("cpu")  # or .cuda() if you want GPU

    x_t = torch.randn(batch_size, size, requires_grad=False)

    start = time.perf_counter()
    for _ in range(iterations):
        model_t.zero_grad(set_to_none=True)
        out_t = model_t(x_t)
        out_t.sum().backward()
    torch_time = (time.perf_counter() - start) / iterations

    return aaron_time, torch_time


# ─── Run ─────────────────────────────────────────────────────────────────────

print(f"{'Dim':<8} | {'Batch':<6} | {'AaronNet (ms)':<14} | {'PyTorch (ms)':<14} | {'Slowdown':<10}")
print("-" * 65)

for dim in [128, 256, 512, 1024]:
    t_a, t_p = benchmark_mlp(size=dim, hidden=dim*2, batch_size=32, iterations=100)
    slowdown = t_a / t_p if t_p > 0 else float('inf')
    print(f"{dim:<8} | {32:<6} | {t_a*1000:12.3f}   | {t_p*1000:12.3f}   | {slowdown:6.1f}x")
