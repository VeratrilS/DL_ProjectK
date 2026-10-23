"""Evaluates the end-to-end Sketch2React pipeline on a labelled dataset split.

Usage:
    python evaluate.py                    # dataset/test
    python evaluate.py --split val --limit 40
"""

import argparse
import os
import time

import numpy as np

from pipeline import UI_CLASSES, get_pipeline


def print_confusion_matrix(cm: np.ndarray):
  width = max(len(c) for c in UI_CLASSES) + 2
  print("true \\ pred".ljust(width) + "".join(c[:8].rjust(10) for c in UI_CLASSES) + "    recall")
  for i, c in enumerate(UI_CLASSES):
    recall = cm[i, i] / max(1, cm[i].sum())
    print(c.ljust(width) + "".join(str(v).rjust(10) for v in cm[i]) + f"    {recall:6.1%}")


def evaluate(data_dir: str, split: str, limit: int = 0):
  pipeline = get_pipeline()
  cm = np.zeros((len(UI_CLASSES), len(UI_CLASSES)), dtype=int)
  unknown = 0
  start = time.time()

  for true_idx, label in enumerate(UI_CLASSES):
    class_dir = os.path.join(data_dir, split, label)
    files = sorted(os.listdir(class_dir))
    if limit:
      files = files[:limit]
    for name in files:
      result = pipeline.process(os.path.join(class_dir, name))
      pred = result["detected_component"]
      if pred in UI_CLASSES:
        cm[true_idx, UI_CLASSES.index(pred)] += 1
      else:
        unknown += 1  # e.g. "Empty Canvas" on a very faint sketch

  total = cm.sum() + unknown
  print(f"\n[Evaluate] {split}: {total} images in {time.time() - start:.1f}s ({unknown} not classified)")
  print(f"[Evaluate] Accuracy: {np.trace(cm) / total:.2%}\n")
  print_confusion_matrix(cm)
  return cm


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", default="dataset")
  parser.add_argument("--split", default="test")
  parser.add_argument("--limit", type=int, default=0, help="max images per class (0 = all)")
  args = parser.parse_args()
  evaluate(args.data, args.split, args.limit)
