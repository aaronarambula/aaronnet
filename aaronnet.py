import numpy as np
from numba import njit

@njit
def njit_matmul_forward(a, b):
    return a @ b

@njit
def njit_relu_forward(a):
    return np.maximum(0.0, a)

@njit 
def njit_add_forward(a, b):
    return a + b

@njit
def njit_matmul_backward(a, b, grad_out):
    grad_a = grad_out @ b.T
    grad_b = a.T @ grad_out

    return grad_a, grad_b

@njit
def njit_relu_backward(data, grad_out):
    # create mask where data was > 0
    mask = (data > 0).astype(np.float32)
    return mask * grad_out

@njit
def njit_softmax_forward(x):
    x_max = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - x_max)
    denom = np.sum(exps, axis=1, keepdims=True)
    return exps / denom

@njit
def njit_softmax_backward(y, dy):
    dot = np.sum(y * dy, axis=1, keepdims=True)
    return y * (dy - dot)

@njit
def njit_adam_step(data, grad, m, v, beta1, beta2, lr, eps, t):
    m[:] = beta1 * m + (1 - beta1) * grad
    v[:] = beta2 * v + (1 - beta2) * (grad**2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    data -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return m, v

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
        out = Tensor(njit_add_forward(self.data, other.data), requires_grad=requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                grad = out.grad
                # broadcasting - sum over dimensions that were broadcasted
                if self.data.shape != grad.shape:
                    # sum over broadcasted dimensions
                    ndims_added = grad.ndim - self.data.ndim
                    for i in range(ndims_added):
                        grad = grad.sum(axis=0)
                    # sum over dimensions where self.data has size 1
                    for i, (dim_orig, dim_grad) in enumerate(
                        zip(self.data.shape, grad.shape)
                    ):
                        if dim_orig == 1 and dim_grad > 1:
                            grad = grad.sum(axis=i, keepdims=True)
                self.grad = (
                    self.grad if self.grad is not None else np.zeros_like(self.data)
                ) + grad
            if other.requires_grad:
                grad = out.grad
                # handle broadcasting for other
                if other.data.shape != grad.shape:
                    ndims_added = grad.ndim - other.data.ndim
                    for i in range(ndims_added):
                        grad = grad.sum(axis=0)
                    for i, (dim_orig, dim_grad) in enumerate(
                        zip(other.data.shape, grad.shape)
                    ):
                        if dim_orig == 1 and dim_grad > 1:
                            grad = grad.sum(axis=i, keepdims=True)

                other.grad = (
                    other.grad if other.grad is not None else np.zeros_like(other.data)
                ) + grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        other = other if isinstance(other, Tensor) else Tensor(other)

        data = njit_matmul_forward(self.data, other.data)
        out = Tensor(data, requires_grad=requires_grad)
        out._prev = {self, other}

        def _backward():
            ga, gb = njit_matmul_backward(self.data, other.data, out.grad)

            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + ga
            if other.requires_grad:
                other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + gb

        out._backward = _backward
        return out
                    
    def relu(self):
        out = Tensor(njit_relu_forward(self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        
        def _backward():
            if self.requires_grad:
                grad = njit_relu_backward(self.data, out.grad)
                self.grad = (self.grad if self.grad is not None else 0) + grad
        out._backward = _backward
        return out

    def softmax(self):
        probs = njit_softmax_forward(self.data)
        out = Tensor(probs, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                #jit gradient calculation
                grad = njit_softmax_backward(out.data, out.grad)
                self.grad = (self.grad if self.grad is not None else 0) + grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        requires_grad = self.requires_grad or (
            isinstance(other, Tensor) and other.requires_grad
        )
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data
                # handle broadcasting
                if self.data.shape != grad.shape:
                    ndims_added = grad.ndim - self.data.ndim
                    for i in range(ndims_added):
                        grad = grad.sum(axis=0)
                    for i, (dim_orig, dim_grad) in enumerate(
                        zip(self.data.shape, grad.shape)
                    ):
                        if dim_orig == 1 and dim_grad > 1:
                            grad = grad.sum(axis=i, keepdims=True)

                self.grad = (
                    self.grad if self.grad is not None else np.zeros_like(self.data)
                ) + grad

            if other.requires_grad:
                grad = out.grad * self.data
                if other.data.shape != grad.shape:
                    ndims_added = grad.ndim - other.data.ndim
                    for i in range(ndims_added):
                        grad = grad.sum(axis=0)
                    for i, (dim_orig, dim_grad) in enumerate(
                        zip(other.data.shape, grad.shape)
                    ):
                        if dim_orig == 1 and dim_grad > 1:
                            grad = grad.sum(axis=i, keepdims=True)

                other.grad = (
                    other.grad if other.grad is not None else np.zeros_like(other.data)
                ) + grad

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def backward():
            if self.requires_grad:
                grad = np.ones_like(self.data) * out.grad
                self.grad = (
                    self.grad if self.grad is not None else np.zeros_like(self.data)
                ) + grad

        out._backward = backward
        return out

    @staticmethod
    def cross_entropy(logits, target_indices):
        m = logits.data.shape[0]

        # compute softmax
        exps = np.exp(logits.data - np.max(logits.data, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=-1, keepdims=True)

        # compute negative log likelihood loss
        log_probs = -np.log(probs[range(m), target_indices] + 1e-10)
        loss_val = np.sum(log_probs) / m

        out = Tensor(loss_val, requires_grad=logits.requires_grad)
        out._prev = {logits}

        def _backward():
            if logits.requires_grad:
                # gradient of (softmax + cross_entropy) = (P - Y)
                grad = probs.copy()
                grad[range(m), target_indices] -= 1
                grad /= m
                grad = grad * out.grad  # chain rule
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

    def step(self, max_grad_norm=None):
        # gradient clipping
        if max_grad_norm is not None:
            total_norm = np.sqrt(
                sum(np.sum(p.grad**2) for p in self.params if p.grad is not None)
            )
            clip_coef = max_grad_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in self.params:
                    if p.grad is not None:
                        p.grad *= clip_coef

        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            self.m[i], self.v[i] = njit_adam_step(
                    p.data, p.grad, self.m[i], self.v[i], self.beta1, self.beta2, self.lr, self.eps, self.t
                    )
    def zero_grad(self):
        for p in self.params:
            p.grad = None

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

class ReLU:
    def __call__(self, x):
        return x.relu()

class Linear:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
                np.random.randn(in_features, out_features) * limit, requires_grad=True
                )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)  
    def __call__(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]

