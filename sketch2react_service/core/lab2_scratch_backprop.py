"""Lab 2: Multi-Layer Neural Network & Backpropagation from Pure Scratch.

Used in Sketch2React for:
- Analytical gradient verification (validating Keras automatic differentiation against first-principles math)
- Pure Python confidence score calibration without external framework overhead
- Demonstrating the complete chain rule in viva/interviews
"""

import math
import random
from typing import List, Tuple


def sigmoid(x: float) -> float:
  """Sigmoid activation function: 1 / (1 + e^(-x))."""
  # Clip x to prevent math.exp overflow
  x_clipped = max(-50.0, min(50.0, x))
  return 1.0 / (1.0 + math.exp(-x_clipped))


def sigmoid_derivative(out: float) -> float:
  """Derivative of Sigmoid in terms of its output: d/dx(sigmoid) = out * (1 - out)."""
  return out * (1.0 - out)


class ScratchMLP:
  """A 3-layer Multi-Layer Perceptron trained using pure Backpropagation and Gradient Descent.

  Architecture: n_input -> n_hidden1 -> n_hidden2 -> n_output
  """

  def __init__(self, n_in: int = 3, n_h1: int = 3, n_h2: int = 3, n_out: int = 2, seed: int = 42):
    random.seed(seed)
    self.n_in = n_in
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.n_out = n_out

    # Initialize weights with small random floats
    self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(n_in)] for _ in range(n_h1)]
    self.b1 = [random.uniform(-0.1, 0.1) for _ in range(n_h1)]

    self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(n_h1)] for _ in range(n_h2)]
    self.b2 = [random.uniform(-0.1, 0.1) for _ in range(n_h2)]

    self.w3 = [[random.uniform(-0.5, 0.5) for _ in range(n_h2)] for _ in range(n_out)]
    self.b3 = [random.uniform(-0.1, 0.1) for _ in range(n_out)]

  def forward(self, x: List[float]) -> Tuple[List[float], List[float], List[float]]:
    """Forward propagation pass: returns (h1, h2, output)."""
    # Hidden Layer 1: z1 = W1 * x + b1; h1 = sigmoid(z1)
    h1 = []
    for i in range(self.n_h1):
      val = self.b1[i] + sum(self.w1[i][j] * x[j] for j in range(self.n_in))
      h1.append(sigmoid(val))

    # Hidden Layer 2: z2 = W2 * h1 + b2; h2 = sigmoid(z2)
    h2 = []
    for i in range(self.n_h2):
      val = self.b2[i] + sum(self.w2[i][j] * h1[j] for j in range(self.n_h1))
      h2.append(sigmoid(val))

    # Output Layer: z3 = W3 * h2 + b3; out = sigmoid(z3)
    out = []
    for i in range(self.n_out):
      val = self.b3[i] + sum(self.w3[i][j] * h2[j] for j in range(self.n_h2))
      out.append(sigmoid(val))

    return h1, h2, out

  def train_step(self, x: List[float], target: List[float], lr: float = 0.1) -> float:
    """Executes a single forward + backward propagation pass for one sample and returns MSE loss."""
    # 1. Forward Pass
    h1, h2, out = self.forward(x)

    # 2. Compute Output Error & Output Delta (delta3)
    # delta3[i] = (target[i] - out[i]) * sigmoid'(out[i])
    error = [target[i] - out[i] for i in range(self.n_out)]
    mse = sum(e * e for e in error) / self.n_out
    delta3 = [error[i] * sigmoid_derivative(out[i]) for i in range(self.n_out)]

    # 3. Backpropagate error to Hidden Layer 2 (delta2)
    # delta2[i] = (sum_j w3[j][i] * delta3[j]) * sigmoid'(h2[i])
    delta2 = []
    for i in range(self.n_h2):
      err_sum = sum(self.w3[j][i] * delta3[j] for j in range(self.n_out))
      delta2.append(err_sum * sigmoid_derivative(h2[i]))

    # 4. Backpropagate error to Hidden Layer 1 (delta1)
    # delta1[i] = (sum_j w2[j][i] * delta2[j]) * sigmoid'(h1[i])
    delta1 = []
    for i in range(self.n_h1):
      err_sum = sum(self.w2[j][i] * delta2[j] for j in range(self.n_h2))
      delta1.append(err_sum * sigmoid_derivative(h1[i]))

    # 5. Gradient Updates (Weight updates = lr * delta * input_activation)
    for i in range(self.n_out):
      for j in range(self.n_h2):
        self.w3[i][j] += lr * delta3[i] * h2[j]
      self.b3[i] += lr * delta3[i]

    for i in range(self.n_h2):
      for j in range(self.n_h1):
        self.w2[i][j] += lr * delta2[i] * h1[j]
      self.b2[i] += lr * delta2[i]

    for i in range(self.n_h1):
      for j in range(self.n_in):
        self.w1[i][j] += lr * delta1[i] * x[j]
      self.b1[i] += lr * delta1[i]

    return mse


# --- Interactive Lab Revision Demo ---
if __name__ == "__main__":
  print("=" * 60)
  print("LAB 2: BACKPROPAGATION & GRADIENT DESCENT FROM PURE SCRATCH")
  print("=" * 60)

  mlp = ScratchMLP(n_in=3, n_h1=3, n_h2=3, n_out=2)
  dummy_inputs = [[0.8, 0.2, 0.1], [0.1, 0.9, 0.8], [0.5, 0.5, 0.0]]
  dummy_targets = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

  print("Training scratch MLP for 500 epochs...")
  for epoch in range(500):
    total_loss = 0.0
    for x, y in zip(dummy_inputs, dummy_targets):
      total_loss += mlp.train_step(x, y, lr=0.2)
    if (epoch + 1) % 100 == 0:
      print(f"Epoch {epoch+1:03d} | Average MSE Loss: {total_loss/len(dummy_inputs):.5f}")

  print("\nInference check after backprop training:")
  for x in dummy_inputs:
    _, _, preds = mlp.forward(x)
    print(f"Input: {x} -> Predicted Output: {[round(p, 3) for p in preds]}")
  print("=" * 60)
