"""Lab 6: Convolutional Neural Networks & Custom Mini-Batch Gradient Descent.

Used in Sketch2React for:
- 2D Spatial Convolution & Pooling for stroke feature extraction
- Custom mini-batch training loop with tf.GradientTape for handling class-imbalanced UI sketches
- Full control over gradient taping, manual batch slicing, and optimizer updates
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_lab6_cnn(input_shape=(28, 28, 1), num_classes: int = 5) -> models.Sequential:
  """Builds the CNN model from Lab 6:

  - Conv2D: 4 filters, (3, 3), activation='relu'
  - MaxPooling2D: (2, 2)
  - Flatten
  - Dense: num_classes, activation='softmax'
  """
  model = models.Sequential([
      layers.Input(shape=input_shape, name="Input_Layer"),
      layers.Conv2D(4, (3, 3), activation="relu", name="Conv2D_Layer"),
      layers.MaxPooling2D((2, 2), name="MaxPool_Layer"),
      layers.Flatten(name="Flatten_Layer"),
      layers.Dense(num_classes, activation="softmax", name="Output_Layer"),
  ], name="Lab6_CNN_Model")

  return model


def custom_minibatch_gradient_descent(
    model: models.Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.01,
) -> dict:
  """Executes the exact Custom Mini-Batch Gradient Descent Loop from Lab 6:

  - Manual index permutation (np.random.permutation) for epoch shuffling
  - Batch slicing with ceil(len / batch_size)
  - Forward pass tracked under tf.GradientTape()
  - tape.gradient(loss, model.trainable_variables)
  - optimizer.apply_gradients()
  """
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

  num_batches = int(np.ceil(len(x_train) / batch_size))
  history = {"loss": [], "accuracy": []}

  start_time = time.time()
  print(f"Training CNN with Custom GradientTape Loop ({epochs} epochs, batch size {batch_size})...")

  for epoch in range(epochs):
    # 1. Shuffling indices each epoch
    indices = np.random.permutation(len(x_train))
    x_shuf = x_train[indices]
    y_shuf = y_train[indices]

    epoch_loss = 0.0
    correct = 0

    for b in range(num_batches):
      X_batch = x_shuf[b * batch_size : (b + 1) * batch_size]
      Y_batch = y_shuf[b * batch_size : (b + 1) * batch_size]
      curr_bsize = len(X_batch)

      # 2. Record forward operations
      with tf.GradientTape() as tape:
        preds = model(X_batch, training=True)
        loss = loss_fn(Y_batch, preds)

      # 3. Compute gradients and apply updates
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      epoch_loss += loss.numpy() * curr_bsize
      correct += np.sum(np.argmax(preds.numpy(), axis=1) == Y_batch)

    avg_loss = epoch_loss / len(x_train)
    accuracy = correct / len(x_train)
    history["loss"].append(avg_loss)
    history["accuracy"].append(accuracy)

    print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

  print(f"Training completed in {time.time() - start_time:.2f}s")
  return history


# --- Interactive Lab Revision Demo ---
if __name__ == "__main__":
  print("=" * 60)
  print("LAB 6: CNN WITH CUSTOM tf.GradientTape MINI-BATCH LOOP")
  print("=" * 60)

  # Generate 200 synthetic sketch patches (28x28 grayscale)
  x_dummy = np.random.uniform(0.0, 1.0, size=(200, 28, 28, 1)).astype(np.float32)
  y_dummy = np.random.randint(0, 5, size=(200,)).astype(np.int32)

  cnn_model = build_lab6_cnn(input_shape=(28, 28, 1), num_classes=5)
  cnn_model.summary()

  hist = custom_minibatch_gradient_descent(cnn_model, x_dummy, y_dummy, epochs=4, batch_size=32)
  print("\nFinal Accuracy:", hist["accuracy"][-1])
  print("=" * 60)
