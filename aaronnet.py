import numpy as np
from numba import njit
from contextlib import contextmanager

# ─── jit kernels ─────────────────────────────────────────────────────────────

@njit
def njit_matmul(a, b):
    return a @ b

@njit
def njit_matmul_backward(a, b, gout):
    ga = gout @ b.T
    gb = a.T @ gout
    return ga, gb

@njit
def njit_relu_forward(x):
    return np.maximum(0.0, x)

@njit
def njit_relu_backward(x, gout):
    return (x > 0).astype(np.float32) * gout

@njit
def njit_softmax_forward(x):
    xmax = np.max(x, axis=1, keepdims=True)
    ex = np.exp(x - xmax)
    return ex / np.sum(ex, axis=1, keepdims=True)

@njit
def njit_softmax_backward(y, dy):  # y = softmax output
    # stable jacobian: diag(y) - y y^T    →   y * (dy - y·dy)
    s = np.sum(y * dy, axis=1, keepdims=True)
    return y * (dy - s)

@njit
def njit_add(a, b):
    return a + b

@njit
def njit_mul(a, b):
    return a * b

@njit
def njit_adam_step(p, g, m, v, beta1, beta2, lr, eps, t):
    m[:] = beta1 * m + (1 - beta1) * g
    v[:] = beta2 * v + (1 - beta2) * (g * g)
    mh = m / (1 - beta1 ** t)
    vh = v / (1 - beta2 ** t)
    p -= lr * mh / (np.sqrt(vh) + eps)


# ─── globals for no_grad ─────────────────────────────────────────────────────

_ENABLE_GRAD = True

@contextmanager
def no_grad():
    global _ENABLE_GRAD
    old = _ENABLE_GRAD
    _ENABLE_GRAD = False
    try:
        yield
    finally:
        _ENABLE_GRAD = old


# ─── Tensor ──────────────────────────────────────────────────────────────────

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad and _ENABLE_GRAD
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self):
        s = f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}"
        if self.grad is not None:
            s += ", has_grad"
        return s + ")"

    @property
    def shape(self):
        return self.data.shape

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float32)

        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(njit_add(self.data, other.data),
                     requires_grad=self.requires_grad or other.requires_grad)

        out._prev = {self, other}

        def _backward():
            gout = out.grad
            if self.requires_grad:
                gs = gout
                if self.shape != gout.shape:
                    axes = tuple(i for i, (a,b) in enumerate(zip(self.shape, gout.shape)) if a == 1 and b > 1)
                    gs = gout.sum(axis=axes, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + gs

            if other.requires_grad:
                go = gout
                if other.shape != gout.shape:
                    axes = tuple(i for i, (a,b) in enumerate(zip(other.shape, gout.shape)) if a == 1 and b > 1)
                    go = gout.sum(axis=axes, keepdims=True)
                other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + go

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(njit_mul(self.data, other.data),
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            gout = out.grad
            if self.requires_grad:
                gs = gout * other.data
                if self.shape != gs.shape:
                    axes = tuple(i for i, (a,b) in enumerate(zip(self.shape, gs.shape)) if a == 1 and b > 1)
                    gs = gs.sum(axis=axes, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + gs

            if other.requires_grad:
                go = gout * self.data
                if other.shape != go.shape:
                    axes = tuple(i for i, (a,b) in enumerate(zip(other.shape, go.shape)) if a == 1 and b > 1)
                    go = go.sum(axis=axes, keepdims=True)
                other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + go

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(njit_matmul(self.data, other.data),
                     requires_grad=self.requires_grad or other.requires_grad)
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
                g = njit_relu_backward(self.data, out.grad)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + g

        out._backward = _backward
        return out

    def softmax(self):
        probs = njit_softmax_forward(self.data)
        out = Tensor(probs, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                g = njit_softmax_backward(out.data, out.grad)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + g

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                g = np.ones_like(self.data) * out.grad
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + g

        out._backward = _backward
        return out

    @staticmethod
    def cross_entropy(logits, targets):  # targets = int indices [0, C-1]
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets, dtype=int)

        m = logits.data.shape[0]
        # log-softmax
        xmax = np.max(logits.data, axis=1, keepdims=True)
        logsumexp = xmax + np.log(np.sum(np.exp(logits.data - xmax), axis=1, keepdims=True))
        log_probs = logits.data - logsumexp
        loss = -np.mean(log_probs[np.arange(m), targets])

        out = Tensor(loss, requires_grad=logits.requires_grad)
        out._prev = {logits}

        def _backward():
            if logits.requires_grad:
                grad = np.exp(log_probs)           # softmax
                grad[np.arange(m), targets] -= 1.0
                grad /= m
                if out.grad is not None:
                    grad *= out.grad
                logits.grad = (logits.grad if logits.grad is not None else np.zeros_like(logits.data)) + grad

        out._backward = _backward
        return out


# ─── Optimizer & Layers ──────────────────────────────────────────────────────

class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, clip_norm=None):
        if clip_norm is not None:
            total_norm = np.sqrt(sum(np.sum(g**2) for p in self.params if p.grad is not None for g in [p.grad]))
            scale = clip_norm / (total_norm + 1e-6)
            if scale < 1:
                for p in self.params:
                    if p.grad is not None:
                        p.grad *= scale

        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            njit_adam_step(p.data, p.grad, self.m[i], self.v[i],
                           self.beta1, self.beta2, self.lr, self.eps, self.t)


class Linear:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(2.0 / in_features)   # He init
        self.weight = Tensor(np.random.randn(in_features, out_features) * limit, requires_grad=True)
        self.bias   = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]


class ReLU:
    def __call__(self, x):
        return x.relu()


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        ps = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                ps.extend(layer.parameters())
        return ps
