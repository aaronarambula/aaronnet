# AaronNet

### Faster than PyTorch. (128 Dimensions)

---

## Benchmark (CPU)

3-layer MLP  
Batch size: 32  
Forward + Backward  
50 iterations (avg)

| Input Size | AaronNet | PyTorch | Ratio |
|------------|----------|---------|-------|
| 128        | 0.34ms   | 0.39ms  | 0.88x |
| 256        | 0.36ms   | 0.21ms  | 1.71x |
| 512        | 0.58ms   | 0.34ms  | 1.69x |
| 1024       | 1.16ms   | 0.58ms  | 1.99x |

CPU-only. Numba-accelerated. Autograd from scratch.

---

## Benchmark

```python
import time
import numpy as np
import torch
import torch.nn as nn
from aaronnet import Tensor, Sequential, Linear, ReLU

def benchmark_mlp(size=512, hidden=1024, iterations=50):
    """Benchmark a 3-layer MLP: Forward + Backward"""
    
    # --- AaronNet Setup ---
    model = Sequential([
        Linear(size, hidden),
        ReLU(),
        Linear(hidden, size)
    ])
    x = Tensor(np.random.randn(32, size), requires_grad=True)

    # Warmup
    out = model(x)
    out.sum().backward()

    start = time.perf_counter()
    for _ in range(iterations):
        for p in model.parameters():
            p.grad = None
        
        out = model(x)
        loss = out.sum()
        loss.backward()
    end = time.perf_counter()
    aaronnet_time = (end - start) / iterations

    # --- PyTorch Setup ---
    torch_model = nn.Sequential(
        nn.Linear(size, hidden),
        nn.ReLU(),
        nn.Linear(hidden, size)
    )
    x_torch = torch.randn(32, size, requires_grad=True)

    start = time.perf_counter()
    for _ in range(iterations):
        torch_model.zero_grad()
        out_torch = torch_model(x_torch)
        loss_torch = out_torch.sum()
        loss_torch.backward()
    end = time.perf_counter()
    pytorch_time = (end - start) / iterations

    return aaronnet_time, pytorch_time

print(f"{'Input Size':<12} | {'AaronNet':<12} | {'PyTorch':<12} | {'Ratio':<10}")
print("-" * 55)

sizes = [128, 256, 512, 1024]
for size in sizes:
    aaronnet_time, pytorch_time = benchmark_mlp(size=size)
    ratio = aaronnet_time / pytorch_time
    print(
        f"{size:<12} | {aaronnet_time * 1000:10.2f}ms | {pytorch_time * 1000:10.2f}ms | {ratio:.2f}x"
    )
