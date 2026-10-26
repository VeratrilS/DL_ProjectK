"""Trains the Lab 7 AlexNet backbone on the sketch dataset and saves the weights.

Steps:
    python generate_dataset.py      # creates dataset/{train,val,test}
    python train.py                 # trains, saves models/alexnet_sketch.weights.h5, reports test accuracy
    python evaluate.py              # evaluates the full end-to-end pipeline
"""

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from core.lab7_alexnet_backbone import (
    DEFAULT_WEIGHTS_PATH,
    SKETCH_INPUT_SHAPE,
    UI_CLASSES,
    build_alexnet_model,
    preprocess_sketch_for_alexnet,
    to_grayscale_227,
)
from evaluate import print_confusion_matrix


def load_split(data_dir: str, split: str, max_per_class: int = 0):
  """Loads one split into (X, y) arrays already preprocessed for AlexNet."""
  images, labels = [], []
  for idx, label in enumerate(UI_CLASSES):
    class_dir = os.path.join(data_dir, split, label)
    names = sorted(os.listdir(class_dir))
    for name in names[:max_per_class or None]:
      gray = cv2.imread(os.path.join(class_dir, name), cv2.IMREAD_GRAYSCALE)
      images.append(preprocess_sketch_for_alexnet(to_grayscale_227(gray)))
      labels.append(idx)
  print(f"[Train] Loaded {split}: {len(images)} images")
  return np.stack(images), np.array(labels, dtype=np.int32)


def main(args):
  tf.random.set_seed(args.seed)
  np.random.seed(args.seed)

  X_train, y_train = load_split(args.data, "train", args.max_per_class)
  X_val, y_val = load_split(args.data, "val")

  model = build_alexnet_model(input_shape=SKETCH_INPUT_SHAPE, num_classes=len(UI_CLASSES), dense_units=256)
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=args.lr),
      loss=keras.losses.SparseCategoricalCrossentropy(),
      metrics=["accuracy"],
  )

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  callbacks = [
      keras.callbacks.ModelCheckpoint(args.out, monitor="val_accuracy", save_best_only=True, save_weights_only=True),
      keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
  ]
  history = model.fit(
      X_train, y_train,
      validation_data=(X_val, y_val),
      epochs=args.epochs,
      batch_size=args.batch_size,
      shuffle=True,
      callbacks=callbacks,
      verbose=2,
  )
  del X_train

  X_test, y_test = load_split(args.data, "test")
  y_pred = np.argmax(model.predict(X_test, batch_size=64, verbose=0), axis=1)
  cm = np.zeros((len(UI_CLASSES), len(UI_CLASSES)), dtype=int)
  for t, p in zip(y_test, y_pred):
    cm[t, p] += 1

  best_epoch = int(np.argmax(history.history["val_accuracy"]))
  print(f"\n[Train] Best epoch {best_epoch + 1}: val_accuracy = {history.history['val_accuracy'][best_epoch]:.2%}")
  print(f"[Train] AlexNet test accuracy: {np.trace(cm) / cm.sum():.2%}\n")
  print_confusion_matrix(cm)
  print(f"\n[Train] Weights saved to {args.out}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", default="dataset")
  parser.add_argument("--out", default=DEFAULT_WEIGHTS_PATH)
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--max-per-class", type=int, default=0, help="cap on training images per class (0 = all); useful on CPU")
  parser.add_argument("--seed", type=int, default=42)
  main(parser.parse_args())
