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
        requires_grad = self.requires_grad or other.requires_grad
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=requires_grad)
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
        requires_grad = self.requires_grad or other.requires_grad
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), requires_grad=requires_grad)
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
        requires_grad = self.requires_grad
        out = Tensor(np.maximum(0, self.data), requires_grad=requires_grad)
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

    @staticmethod
    def cross_entropy(logits, target_indices):
        m = logits.data.shape[0]

        # compute softmax
        exps = np.exp(logits.data - np.max(logits.data, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        # compute negative log likelihood loss
        log_probs = -np.log(probs[range(m), target_indices])
        loss_val = np.sum(log_probs) / m

        out = Tensor(loss_val, requires_grad=logits.requires_grad)
        out._prev = {logits}

        def _backward():
            if logits.requires_grad:
                # gradient of (softmax + cross_entropy) = (P - Y)
                grad = probs.copy()
                grad[range(m), target_indices] -= 1
                grad /= m
                logits.grad = (
                    logits.grad
                    if logits.grad is not None
                    else np.zeros_like(logits.data)
                ) + grad

        out._backward = _backward
        return out


class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
