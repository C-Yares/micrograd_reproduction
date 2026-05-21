# micrograd-scratch

A scalar-valued autograd engine and tiny neural-network library, built from scratch in Python. Implements reverse-mode automatic differentiation, verified against PyTorch to floating-point precision, and trains a small MLP end-to-end.

Built as a self-contained exercise after watching Andrej Karpathy's [micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0) — re-implemented from scratch without referencing the original [`engine.py`](https://github.com/karpathy/micrograd), to verify I'd actually internalized the material.

---

## Quick demo

```python
from micrograd.engine import Value

x = Value(2.0)
y = Value(3.0)
z = x * y + x          # z = xy + x = 8
z.backward()

print(x.grad)   # 4.0   (∂z/∂x = y + 1)
print(y.grad)   # 2.0   (∂z/∂y = x)
```

Three lines build a computation graph; `z.backward()` walks it in reverse topological order and applies the chain rule at every node.

---

## What's in this repo

```
micrograd-scratch/
├── micrograd/
│   ├── __init__.py
│   ├── engine.py      # the Value class + autograd engine
│   ├── nn.py          # Neuron, Layer, MLP built on top of Value
│   └── trial.py       # scratch tests written during development
└── tests/
    ├── test_engine.py # PyTorch verification — passes
    └── demo_train.py  # tiny MLP training loop — converges
```

### `engine.py` — the autograd engine

A `Value` class that wraps a single float and, when used in arithmetic, builds a computation graph. Each `Value` stores:

- Its numeric `data`
- A `grad` field (initialized to 0)
- A `_prev` set of parent `Value`s
- A `_backward` closure: given the gradient flowing in via `out.grad`, computes and accumulates gradients for the parents

Supported operations: `+`, `-`, `*`, `/`, unary `-`, `**` (scalar exponent), `relu`. Reverse operators (`__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`) are also implemented so that expressions like `2 * Value(3)` work.

The `backward()` method:
1. Builds a topological ordering of all nodes reachable from `self`
2. Sets `self.grad = 1.0` (the seed: $\partial L/\partial L = 1$)
3. Walks the topo order in reverse, calling each node's `_backward` closure to propagate gradients to its parents

### `nn.py` — the tiny neural-net library

Three classes layered on top of `Value`, plus a small `Module` base for shared methods:

- `Neuron(nin, nonlin=True)` — `nin` random weights in [-1, 1], a bias, optional ReLU
- `Layer(nin, nout, **kwargs)` — a list of `Neuron`s
- `MLP(nin, nouts)` — a list of `Layer`s; e.g. `MLP(3, [4, 4, 1])` is a 3-input network with two ReLU hidden layers of 4 and a linear output

All inherit a `parameters()` method that flattens trainable `Value`s, and a `zero_grad()` method that clears gradients before each backward pass.

### `tests/test_engine.py` — PyTorch verification

Builds a nontrivial expression involving `+`, `*`, `relu`, and constant arithmetic — once in micrograd, once in PyTorch — then asserts forward values and gradients agree to within $10^{-6}$:

```bash
$ python tests/test_engine.py
PASS: forward and backward match PyTorch
```

### `tests/demo_train.py` — end-to-end training

A 3-input MLP with two hidden layers of 4 (ReLU) and a single linear output, trained for 100 steps of SGD on a 4-example toy dataset with mean-squared loss:

```bash
$ python tests/demo_train.py
step 0:  loss = 5.381633
step 10: loss = 3.422688
step 20: loss = 0.199859
step 30: loss = 0.091827
step 40: loss = 0.037322
step 50: loss = 0.011737
step 60: loss = 0.002974
step 70: loss = 0.000657
step 80: loss = 0.000139
step 90: loss = 0.000030
predictions: [1.0014, -0.9992, -1.0010, 0.9981]
targets:     [1.0,    -1.0,    -1.0,    1.0]
```

Loss drops by five orders of magnitude over 100 steps; predictions converge to within 0.2% of targets. Confirms the full forward → backward → optimizer step loop is working end-to-end on a real learning problem.

---

## What I learned by building this

Re-implementing from scratch (after watching the lecture, but without referring back to the original `engine.py`) forced me to confront several things I'd glossed over.

**1. Reverse-mode autodiff is the chain rule applied to a graph — nothing more.** Every "magical" thing about `loss.backward()` in PyTorch decomposes into: topologically sort the operations, walk in reverse, multiply by local Jacobians at each step. The core `backward()` method is about 15 lines. Everything else (operators, neural-net wrappers, training loops) is infrastructure built around it.

**2. Every operation needs a local derivative rule.** Each backward closure implements one line of the chain rule: `parent.grad += (local derivative) * out.grad`. For addition the local derivative is `1`; for multiplication it's the *other* input; for power it's `n * x^(n-1)`; for ReLU it's `1 if x>0 else 0`. PyTorch's autograd is the same idea, just with thousands of these rules written in C++/CUDA.

**3. Gradient accumulation via `+=` is essential.** When a `Value` is used in multiple downstream operations, each use contributes a term to its gradient. `+=` makes them sum (multivariable chain rule). I had this exact bug in my initial `__pow__` implementation — I wrote `self.grad = ...` instead of `+=`, which silently overwrote contributions from other paths. Fixing it taught me *why* every backward function must accumulate.

**4. Reverse operators need explicit `return`s.** Another bug I shipped briefly: `__sub__` and `__neg__` computed the right values but had no `return` statement, so they implicitly returned `None`. Standard `__add__` worked, so my PyTorch test passed by luck of which operators it happened to hit. The bug only surfaced when I tried to subtract `Value`s in the training loop. Lesson: a passing test only proves the parts it tests, not the parts it doesn't.

**5. Learning rate tuning is real, even on tiny models.** My first training run used `lr=0.05` — the loss dropped fast to ~0.04, then bounced back to 2.8. Classic learning-rate-too-high pattern: large steps overshoot the minimum once you're close. Dropping to `lr=0.03` gave the smooth monotonic curve above. You don't need a transformer to feel this — it shows up at every scale.

**6. Why production frameworks moved to tensors.** Each `Value` is one graph node holding one number. A 41-parameter MLP creates hundreds of `Value`s per forward pass — slow in Python. Production frameworks (PyTorch, JAX) operate on tensors so that one matmul is one node, even when processing millions of numbers. Micrograd is intentionally pedagogical; its performance limit is the price of its clarity.

---

## How to run

```bash
git clone https://github.com/yourusername/micrograd-scratch
cd micrograd-scratch
pip install torch numpy
python tests/test_engine.py    # PyTorch verification
python tests/demo_train.py     # tiny MLP training
```

Requires: Python 3.10+, NumPy, PyTorch (for the verification test only).

---

## What's next

This is the first hands-on project in my summer plan toward understanding deep learning from first principles. Next:

- 2-layer MLP from scratch in NumPy for MNIST (explicit batched backprop, no autograd)
- Karpathy's `makemore` series (bigram → MLP → BatchNorm → manual backprop)
- Eventually: a small transformer in the `nanoGPT` style

Micrograd was the foundation. Once you understand backprop at this level, deep learning frameworks stop feeling like magic — they're just this same machinery scaled up with tensors and GPUs.

---

## Acknowledgments

All credit to Andrej Karpathy for the [original micrograd](https://github.com/karpathy/micrograd) and the [accompanying lecture](https://www.youtube.com/watch?v=VMj-3S1tku0) — one of the best teaching resources in ML. This implementation is my own work, written after watching the lecture without referring back to the original `engine.py`. PyTorch tests confirm the gradients are correct, and a tiny end-to-end training run confirms the full library works.