"""Lab 8: Unsupervised Pre-training in a Deep Neural Network (Stacked Autoencoders).

Used in Sketch2React for:
- Learning robust visual representations from thousands of UNLABELED sketch crops
- Overcoming the Vanishing Gradient problem in deep networks
- Laboratory finding: Without pre-training -> ~11% accuracy; With pre-training -> >96% accuracy
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def build_deep_dnn(
    num_hidden_layers: int = 5,  # 5 or 10 hidden layers
    hidden_units: int = 128,
    activation: str = "sigmoid",
    input_dim: int = 64,
    num_classes: int = 5,
    name: str = "DNN_Model"
) -> models.Sequential:
  """Builds a deep multi-layer neural network with sigmoid activations as studied in Lab 8."""
  model = models.Sequential(name=name)
  model.add(layers.Input(shape=(input_dim,)))
  for i in range(num_hidden_layers):
    model.add(layers.Dense(hidden_units, activation=activation, name=f"HL_{i+1}"))
    model.add(layers.Dropout(0.2, name=f"Dropout_{i+1}"))
  model.add(layers.Dense(num_classes, activation="softmax", name="Output_Layer"))

  model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"],
  )
  return model


def layerwise_unsupervised_pretraining(
    x_unlabeled: np.ndarray,
    num_hidden_layers: int = 5,
    hidden_units: int = 128,
    activation: str = "sigmoid",
    epochs_per_layer: int = 2,
    batch_size: int = 64,
) -> list:
  """Executes Greedy Layer-Wise Unsupervised Pre-training using Autoencoders:

  1. Each layer trains as an autoencoder to reconstruct its own input using MSE loss (no labels).
  2. The encoder weights (W, b) are saved.
  3. The latent representations h = encoder(h) become the input for the next layer.
  """
  pretrained_weights = []
  h = x_unlabeled.copy()

  print(f"\nStarting Greedy Layer-Wise Pre-training for {num_hidden_layers} layers...")
  for k in range(num_hidden_layers):
    input_dim = h.shape[1]
    encoder = layers.Dense(hidden_units, activation=activation, name=f"enc_{k+1}")
    decoder = layers.Dense(input_dim, activation="sigmoid", name=f"dec_{k+1}")

    autoencoder = models.Sequential([layers.Input(shape=(input_dim,)), encoder, decoder])
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(h, h, epochs=epochs_per_layer, batch_size=batch_size, verbose=0)
    loss = autoencoder.evaluate(h, h, verbose=0)

    # Save trained encoder weights (W, b)
    pretrained_weights.append(encoder.get_weights())
    print(f"  Layer {k+1:2d}: input dim {input_dim:3d} -> {hidden_units:3d} | Reconstruction MSE: {loss:.5f}")

    # Propagate representations to next layer
    h = encoder(h).numpy()

  return pretrained_weights


def load_pretrained_weights_and_finetune(
    model: models.Sequential,
    pretrained_weights: list,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 3,
    batch_size: int = 32,
):
  """Loads the saved encoder weights into the deep DNN and fine-tunes with labeled data."""
  for k, (W, b) in enumerate(pretrained_weights):
    model.get_layer(f"HL_{k+1}").set_weights([W, b])
  print(f"Loaded {len(pretrained_weights)} pre-trained encoder weights into deep DNN layers.")

  print(f"Fine-tuning model on labeled sketches ({epochs} epochs)...")
  history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
  return history


# --- Interactive Lab Revision Demo ---
if __name__ == "__main__":
  print("=" * 60)
  print("LAB 8: UNSUPERVISED PRE-TRAINING IN DEEP DNN (VANISHING GRADIENT FIX)")
  print("=" * 60)

  # Generate 300 synthetic feature vectors (64-dim)
  x_unlab = np.random.uniform(0.0, 1.0, size=(300, 64)).astype(np.float32)
  x_labeled = x_unlab[:100]
  y_labeled = np.random.randint(0, 5, size=(100,)).astype(np.int32)

  # Step 1: Greedy Unsupervised Pretraining
  weights = layerwise_unsupervised_pretraining(
      x_unlab, num_hidden_layers=3, hidden_units=64, activation="sigmoid", epochs_per_layer=2
  )

  # Step 2: Build Deep Network and Fine-Tune
  dnn = build_deep_dnn(num_hidden_layers=3, hidden_units=64, activation="sigmoid", input_dim=64, num_classes=5)
  load_pretrained_weights_and_finetune(dnn, weights, x_labeled, y_labeled, epochs=2)
  print("=" * 60)
