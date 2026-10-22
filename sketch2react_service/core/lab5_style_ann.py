"""Lab 5: Train an ANN with Dropout Regularization.

Used in Sketch2React for:
- Component Variant & Styling Classification (e.g. Primary, Secondary, Outline, Danger styles)
- Regularizing dense representations with Dropout (p = 0.2) to prevent overfitting on hand-drawn quirks
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_lab5_style_ann(input_shape=(28, 28), num_classes: int = 10) -> models.Sequential:
  """Builds the exact Multi-Layer Perceptron architecture from Lab 5:

  - Input Layer: Flatten (28x28 -> 784)
  - First H/L: 16 neurons with sigmoid activation
  - Dropout: probability = 0.2
  - Second H/L: 8 neurons with sigmoid activation
  - Dropout: probability = 0.2
  - Output Layer: num_classes neurons with softmax activation
  """
  model = models.Sequential([
      layers.Input(shape=input_shape),
      layers.Flatten(name="Input_Flatten"),
      layers.Dense(16, activation="sigmoid", name="First_HL"),
      layers.Dropout(0.2, name="Dropout_1"),
      layers.Dense(8, activation="sigmoid", name="Second_HL"),
      layers.Dropout(0.2, name="Dropout_2"),
      layers.Dense(num_classes, activation="softmax", name="Output_Layer"),
  ], name="Lab5_Style_ANN")

  model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"],
  )
  return model


def predict_component_style(model: models.Sequential, feature_vector_28x28: np.ndarray) -> dict:
  """Predicts the React component style variant from extracted bounding box stroke patch."""
  # Style classes for React components
  STYLE_CLASSES = ["Primary Filled", "Secondary Ghost", "Outline Bordered", "Danger Alert", "Subtle Minimal"]
  
  sample = np.expand_dims(feature_vector_28x28, axis=0)
  probs = model(sample, training=False).numpy()[0]
  best_idx = int(np.argmax(probs)) % len(STYLE_CLASSES)

  return {
      "predicted_variant": STYLE_CLASSES[best_idx],
      "confidence": f"{probs[best_idx] * 100:.2f}%",
      "style_probabilities": [round(float(p), 4) for p in probs[:len(STYLE_CLASSES)]]
  }


# --- Interactive Lab Revision Demo ---
if __name__ == "__main__":
  print("=" * 60)
  print("LAB 5: MULTI-LAYER PERCEPTRON WITH DROPOUT (0.2)")
  print("=" * 60)

  model = build_lab5_style_ann(input_shape=(28, 28), num_classes=10)
  print("\nModel Architecture Summary:")
  model.summary()

  # Quick synthetic training demonstration
  dummy_x = np.random.uniform(0.0, 1.0, size=(100, 28, 28)).astype(np.float32)
  dummy_y = np.random.randint(0, 10, size=(100,)).astype(np.int32)

  print("\nTraining for 3 epochs with Adam & Sparse Categorical Crossentropy...")
  history = model.fit(dummy_x, dummy_y, epochs=3, batch_size=32, verbose=1)

  # Style prediction demo
  res = predict_component_style(model, dummy_x[0])
  print("\nStyle Prediction on Sample Sketch Patch:")
  print(res)
  print("=" * 60)
