
from aaronnet import Sequential, Linear, ReLU, AdamOptimizer, Tensor
import numpy as np

model = Sequential([
    Linear(512, 256),
    ReLU(),
    Linear(256, 10)
])

x = Tensor(np.random.randn(64, 512), requires_grad=True)
y = Tensor(np.random.randint(0, 10, size=(64,)))

logits = model(x)
loss = Tensor.cross_entropy(logits, y.data)
loss.backward()

opt = AdamOptimizer(model.parameters(), lr=0.001)
opt.step()
