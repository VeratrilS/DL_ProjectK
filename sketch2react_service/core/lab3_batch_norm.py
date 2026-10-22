"""Lab 3: Batch Normalization — Scratch (NumPy) vs Standard Library (TensorFlow).

Used in Sketch2React for:
- Standardizing activation feature maps across hand-drawn sketches with varying pen thickness, pressure, and background contrast
- Preventing internal covariate shift so deep layers train stably
"""

import numpy as np
import tensorflow as tf


def batch_norm_scratch(x: np.ndarray, gamma: float = 1.0, beta: float = 0.0, eps: float = 1e-5) -> np.ndarray:
  """Implements Batch Normalization from first principles without using standard libraries.

  Equations:
      mu    = 1/m * sum(x_i)
      var   = 1/m * sum((x_i - mu)^2)
      x_hat = (x - mu) / sqrt(var + eps)
      out   = gamma * x_hat + beta
  """
  mean = np.mean(x, axis=0)
  var = np.var(x, axis=0)
  x_norm = (x - mean) / np.sqrt(var + eps)
  return gamma * x_norm + beta


def relu_scratch(x: np.ndarray) -> np.ndarray:
  """ReLU activation function from scratch."""
  return np.maximum(0, x)


def build_tf_batch_norm_network(input_dim: int = 3, gamma_val: float = 0.2, beta_val: float = 1.3, eps: float = 0.01) -> tf.keras.Model:
  """Implements the exact 4-layer ANN with BatchNormalization from Lab 3 using TensorFlow/Keras."""
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(input_dim,)),
      tf.keras.layers.Dense(3, use_bias=False),
      tf.keras.layers.BatchNormalization(
          epsilon=eps,
          gamma_initializer=tf.keras.initializers.Constant(gamma_val),
          beta_initializer=tf.keras.initializers.Constant(beta_val),
      ),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.Dense(3, use_bias=False),
      tf.keras.layers.BatchNormalization(
          epsilon=eps,
          gamma_initializer=tf.keras.initializers.Constant(gamma_val),
          beta_initializer=tf.keras.initializers.Constant(beta_val),
      ),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.Dense(2, use_bias=False),
      tf.keras.layers.BatchNormalization(
          epsilon=eps,
          gamma_initializer=tf.keras.initializers.Constant(gamma_val),
          beta_initializer=tf.keras.initializers.Constant(beta_val),
      ),
      tf.keras.layers.Activation("relu"),
  ], name="Lab3_BatchNorm_Network")

  return model


# --- Interactive Lab Revision Demo ---
if __name__ == "__main__":
  print("=" * 60)
  print("LAB 3: BATCH NORMALIZATION (SCRATCH VS KERAS)")
  print("=" * 60)

  # Sample mini-batch of 3 samples, 3 features each
  X = np.array([
      [2.3, -1.9, 7.2],
      [1.5, 0.4, 3.1],
      [-0.8, 2.1, 5.0]
  ], dtype=np.float32)

  gamma = 0.2
  beta = 1.3
  eps = 0.01

  # 1. Scratch Batch Normalization
  out_scratch = batch_norm_scratch(X, gamma=gamma, beta=beta, eps=eps)
  print("\n1. NumPy Scratch Batch Normalization Output:\n", out_scratch)
  print(f"Normalized Mean (should be near 0): {np.mean((X - np.mean(X, axis=0)) / np.sqrt(np.var(X, axis=0) + eps), axis=0)}")

  # 2. TensorFlow Keras Batch Normalization
  bn_model = build_tf_batch_norm_network(input_dim=3, gamma_val=gamma, beta_val=beta, eps=eps)
  out_tf = bn_model(X, training=True).numpy()
  print("\n2. TensorFlow Model Output with BatchNorm:\n", out_tf)
  print("=" * 60)
