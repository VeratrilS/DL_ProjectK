"""Lab 7: AlexNet Architecture from Scratch & Custom Local Response Normalization.

Used in Sketch2React for:
- Primary visual backbone: Classifying hand-drawn wireframes into React UI component classes
- Custom LRN layer subclass for channel-wise lateral inhibition
- Overlapping pooling (3x3, stride 2) and heavy Dropout (0.5) to prevent overfitting
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LocalResponseNormalization(layers.Layer):
  """Custom Local Response Normalization (LRN) layer from Lab 7.

  Subclasses tf.keras.layers.Layer using tf.nn.local_response_normalization.
  Formula:
      b_{x,y}^i = a_{x,y}^i / (bias + alpha * sum_{j}(a_{x,y}^j)^2)^beta
  """

  def __init__(self, depth_radius: int = 2, bias: float = 2.0, alpha: float = 1e-4, beta: float = 0.75, **kwargs):
    super().__init__(**kwargs)
    self.depth_radius = depth_radius
    self.bias = bias
    self.alpha = alpha
    self.beta = beta

  def call(self, inputs):
    return tf.nn.local_response_normalization(
        inputs,
        depth_radius=self.depth_radius,
        bias=self.bias,
        alpha=self.alpha,
        beta=self.beta,
    )

  def get_config(self):
    config = super().get_config()
    config.update({
        "depth_radius": self.depth_radius,
        "bias": self.bias,
        "alpha": self.alpha,
        "beta": self.beta,
    })
    return config


def build_alexnet_model(
    input_shape=(227, 227, 3),
    num_classes: int = 5,
    dense_units: int = 512,  # Compact 512 for efficient local microservice execution
) -> keras.Model:
  """Builds the AlexNet Deep CNN Architecture as learned in Lab 7:

  1. Conv1 (11x11, stride 4, 96 filters) -> LRN1 -> Pool1 (3x3, stride 2)
  2. Conv2 (5x5, padding same, 256 filters) -> LRN2 -> Pool2 (3x3, stride 2)
  3. Conv3 (3x3, padding same, 384 filters)
  4. Conv4 (3x3, padding same, 384 filters)
  5. Conv5 (3x3, padding same, 256 filters) -> Pool3 (3x3, stride 2)
  6. FC1 (dense_units) -> Dropout(0.5)
  7. FC2 (dense_units) -> Dropout(0.5)
  8. FC3 (num_classes, softmax)
  """
  inputs = keras.Input(shape=input_shape, name="sketch_image_input")

  # Block 1
  x = layers.Conv2D(96, kernel_size=(11, 11), strides=4, activation="relu", name="conv1")(inputs)
  x = LocalResponseNormalization(name="lrn1")(x)
  x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, name="pool1")(x)

  # Block 2
  x = layers.Conv2D(256, kernel_size=(5, 5), padding="same", activation="relu", name="conv2")(x)
  x = LocalResponseNormalization(name="lrn2")(x)
  x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, name="pool2")(x)

  # Block 3, 4, 5
  x = layers.Conv2D(384, kernel_size=(3, 3), padding="same", activation="relu", name="conv3")(x)
  x = layers.Conv2D(384, kernel_size=(3, 3), padding="same", activation="relu", name="conv4")(x)
  x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu", name="conv5")(x)
  x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, name="pool5")(x)

  # Classifier Head
  x = layers.Flatten(name="flatten")(x)
  x = layers.Dense(dense_units, activation="relu", name="fc1")(x)
  x = layers.Dropout(0.5, name="dropout1")(x)
  x = layers.Dense(dense_units, activation="relu", name="fc2")(x)
  x = layers.Dropout(0.5, name="dropout2")(x)
  outputs = layers.Dense(num_classes, activation="softmax", name="fc3_output")(x)

  model = keras.Model(inputs=inputs, outputs=outputs, name="AlexNet_Sketch2React")
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-4),
      loss=keras.losses.SparseCategoricalCrossentropy(),
      metrics=["accuracy"],
  )
  return model


def generate_synthetic_ui_dataset(num_samples: int = 100, img_size: int = 227):
  """Generates synthetic wireframe sketches of UI primitives:

  0: Button, 1: Card, 2: InputField, 3: Navbar, 4: Modal
  """
  images = np.ones((num_samples, img_size, img_size, 3), dtype=np.float32) * 0.95
  labels = np.random.randint(0, 5, size=(num_samples,), dtype=np.int32)

  # Add simulated black hand-drawn bounding box strokes
  for i in range(num_samples):
    lbl = labels[i]
    if lbl == 0:  # Button: elongated horizontal rectangle
      images[i, 90:130, 40:180, :] = 0.1
    elif lbl == 1:  # Card: tall/square rounded rectangle
      images[i, 30:190, 40:180, :] = 0.2
    elif lbl == 2:  # InputField: thin line with placeholder
      images[i, 100:125, 30:190, :] = 0.3
    elif lbl == 3:  # Navbar: top full-width bar
      images[i, 10:45, 10:217, :] = 0.15
    elif lbl == 4:  # Modal: centered overlay box
      images[i, 50:170, 50:170, :] = 0.25

  return images, labels


# --- Interactive Lab Revision Demo ---
if __name__ == "__main__":
  print("=" * 60)
  print("LAB 7: ALEXNET ARCHITECTURE FROM SCRATCH (WITH CUSTOM LRN)")
  print("=" * 60)

  # 1. Test custom LRN layer
  lrn = LocalResponseNormalization()
  test_tensor = tf.random.uniform((1, 4, 4, 8), 0, 50)
  normed = lrn(test_tensor)
  print(f"LRN Input Mean:  {float(tf.reduce_mean(test_tensor)):.4f}")
  print(f"LRN Output Mean: {float(tf.reduce_mean(normed)):.4f}")
  print(f"Mean |Input - Output|: {float(tf.reduce_mean(tf.abs(test_tensor - normed))):.4f}")

  # 2. Build AlexNet
  alexnet = build_alexnet_model(input_shape=(227, 227, 3), num_classes=5, dense_units=256)
  print("\nAlexNet Model Summary:")
  alexnet.summary()

  # 3. Train on synthetic sketches
  X_syn, y_syn = generate_synthetic_ui_dataset(num_samples=40, img_size=227)
  print("\nTraining on sample synthetic UI sketches for 2 epochs...")
  alexnet.fit(X_syn, y_syn, epochs=2, batch_size=16, verbose=1)
  print("=" * 60)
