"""Lab 1: TensorFlow Computation Graphs & Basic Keras Sequential Model.

Used in Sketch2React for:
- Converting raw image stroke arrays into immutable TensorFlow tensors (tf.constant)
- Tensor arithmetic (tf.add, tf.subtract) for normalizing pixel ranges and brightness
- Building the underlying Keras Sequential execution graph for component embedding
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential


def tensor_arithmetic_demo(val1: float, val2: float):
  """Demonstrates basic tensor arithmetic using tf.constant, tf.add, tf.subtract."""
  t1 = tf.constant(val1, dtype=tf.float32)
  t2 = tf.constant(val2, dtype=tf.float32)

  added = tf.add(t1, t2)
  subtracted = tf.subtract(t1, t2)

  return float(added.numpy()), float(subtracted.numpy())


def build_basic_sequential_ann(input_dim: int = 4, hidden_units: int = 8, output_classes: int = 5) -> Sequential:
  """Builds, compiles, and returns a standard Keras Sequential Model as learned in Lab 1."""
  model = Sequential([
      Input(shape=(input_dim,), name="feature_input"),
      Dense(hidden_units, activation="relu", name="hidden_layer"),
      Dense(output_classes, activation="softmax", name="output_layer"),
  ], name="Lab1_Sequential_Model")

  model.compile(
      optimizer="adam",
      loss="categorical_crossentropy",
      metrics=["accuracy"],
  )
  return model


# --- Interactive Lab Revision Demo ---
if __name__ == "__main__":
  print("=" * 60)
  print("LAB 1: TENSORFLOW BASICS & SEQUENTIAL MODEL")
  print("=" * 60)

  # 1. Tensor arithmetic
  add_res, sub_res = tensor_arithmetic_demo(10.5, 4.2)
  print(f"Tensor Addition (10.5 + 4.2)      = {add_res:.2f}")
  print(f"Tensor Subtraction (10.5 - 4.2)   = {sub_res:.2f}")

  # 2. Sequential Model summary
  ann = build_basic_sequential_ann(input_dim=4, hidden_units=8, output_classes=5)
  print("\nModel Architecture Summary:")
  ann.summary()
  print("=" * 60)
