import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def backward(self, grad=None):
        # topological order of graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Initialize gradient for output node
        if grad is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = grad

        # go through list in reverse
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = (
                    self.grad if self.grad is not None else np.zeros_like(self.data)
                ) + out.grad
            if other.requires_grad:
                other.grad = (
                    other.grad if other.grad is not None else np.zeros_like(other.data)
                ) + out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data))
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = (
                    self.grad if self.grad is not None else np.zeros_like(self.data)
                ) + np.matmul(out.grad, other.data.T)

            if other.requires_grad:
                other.grad = (
                    other.grad if other.grad is not None else np.zeros_like(other.data)
                ) + np.matmul(self.data.T, out.grad)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data))
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                mask = (self.data > 0).astype(np.float32)
                self.grad = (
                    self.grad if self.grad is not None else np.zeros_like(self.data)
                ) + mask * out.grad

        out._backward = _backward
        return out

    def softmax(self):
        exps = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        out = Tensor(probs, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                self.grad = (
                    self.grad if self.grad is not None else np.zeros_like(self.data)
                )
                for i, (p, g) in enumerate(zip(out.data, out.grad)):
                    p = p.reshape(-1, 1)
                    jacobian = np.diagflat(p) - np.dot(p, p.T)
                    self.grad[i] += np.dot(jacobian, g)

        out._backward = _backward
        return out

    def cross_entropy(logits, target_indices):
        m = logits.data.shape[0]

        # compute softmax
        exps = np.exp(logits.data - np.max(logits.data, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        # compute negative log likelihood loss
        log_probs = -np.log(probs[range(m), target_indices])
        loss_val = np.sum(log_probs) / m

        out = Tensor(loss_val, requires_grad=True)
        out._prev = {logits}

        def _backward():
            if logits.requires_grad:
                # gradient of (softmax + cross_entropy) = (P - Y)
                grad = probs.copy()
                grad[range(m), target_indices] -= 1
                grad /= m
                logits.grad = (logits.grad if logits.grad is not None else 0) + grad

        out._backward = _backward
        return out
