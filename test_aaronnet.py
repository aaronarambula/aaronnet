from aaronnet import Tensor, AdamOptimizer
import numpy as np

# inputs & parameters
x = Tensor(np.random.randn(10, 5), requires_grad=True)
w = Tensor(np.random.randn(5, 3), requires_grad=True)
b = Tensor(np.zeros(3), requires_grad=True)
y_true = np.random.randint(0, 3, size=(10,))

optimizer = AdamOptimizer([w, b], lr=0.01)

# training loop
epochs = 50

for epoch in range(epochs):
    # forward pass
    logits = x @ w + b
    loss = Tensor.cross_entropy(logits, y_true)

    # backward pass
    loss.backward()

    # output progress
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Loss = {loss.data:.4f}")
        print(f"Grad sample (w[0]): {w.grad[0]}")

    # update parameters
    optimizer.step()

    # zero gradients for next step
    optimizer.zero_grad()

# to get final loss and gradients, recompute last forward and backward pass
logits = x @ w + b
loss = Tensor.cross_entropy(logits, y_true)
loss.backward()

print("Final loss: ", loss.data)
print("Final w.grad sample: ", w.grad[0])
print("Final b.grad sample: ", b.grad[0])
