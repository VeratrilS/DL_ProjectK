"""Generates realistic hand-drawn UI sketches and saves them to disk.

Simulates natural hand-drawn stroke jitter, varied pen pressure, and pencil shading.
"""

import os
import random
import cv2
import numpy as np


def add_sketchy_line(canvas, pt1, pt2, color=(30, 30, 30), thickness=2, jitter=2):
  """Draws a hand-drawn wobbly line between two points."""
  x1, y1 = pt1
  x2, y2 = pt2
  num_segments = max(4, int(np.hypot(x2 - x1, y2 - y1) / 15))

  xs = np.linspace(x1, x2, num_segments)
  ys = np.linspace(y1, y2, num_segments)

  for i in range(len(xs) - 1):
    jx1 = int(xs[i] + random.randint(-jitter, jitter))
    jy1 = int(ys[i] + random.randint(-jitter, jitter))
    jx2 = int(xs[i + 1] + random.randint(-jitter, jitter))
    jy2 = int(ys[i + 1] + random.randint(-jitter, jitter))
    cv2.line(canvas, (jx1, jy1), (jx2, jy2), color, thickness)


def add_sketchy_rect(canvas, top_left, bottom_right, color=(35, 35, 35), thickness=2, jitter=2):
  """Draws a hand-drawn rectangle with wobbly lines."""
  x1, y1 = top_left
  x2, y2 = bottom_right
  add_sketchy_line(canvas, (x1, y1), (x2, y1), color, thickness, jitter)
  add_sketchy_line(canvas, (x2, y1), (x2, y2), color, thickness, jitter)
  add_sketchy_line(canvas, (x2, y2), (x1, y2), color, thickness, jitter)
  add_sketchy_line(canvas, (x1, y2), (x1, y1), color, thickness, jitter)


def create_sample_sketches(output_dir="sample_sketches"):
  os.makedirs(output_dir, exist_ok=True)
  random.seed(42)

  # 1. Button Sketch
  btn_canvas = np.ones((227, 227, 3), dtype=np.uint8) * 248
  # Background paper texture
  noise = np.random.normal(0, 3, (227, 227, 3)).astype(np.int16)
  btn_canvas = np.clip(btn_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
  add_sketchy_rect(btn_canvas, (40, 85), (185, 140), thickness=3, jitter=2)
  cv2.putText(btn_canvas, "Sign Up", (68, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
  btn_path = os.path.join(output_dir, "button_sketch.png")
  cv2.imwrite(btn_path, btn_canvas)

  # 2. Card Sketch
  card_canvas = np.ones((227, 227, 3), dtype=np.uint8) * 248
  card_canvas = np.clip(card_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
  add_sketchy_rect(card_canvas, (30, 20), (195, 205), thickness=3, jitter=2)
  # Image placeholder inside card
  add_sketchy_rect(card_canvas, (45, 35), (180, 100), thickness=2, jitter=1)
  add_sketchy_line(card_canvas, (45, 35), (180, 100), thickness=1, jitter=1)
  add_sketchy_line(card_canvas, (180, 35), (45, 100), thickness=1, jitter=1)
  # Title & description lines
  add_sketchy_line(card_canvas, (45, 120), (160, 120), thickness=2, jitter=1)
  add_sketchy_line(card_canvas, (45, 138), (140, 138), thickness=1, jitter=1)
  # Mini button inside card
  add_sketchy_rect(card_canvas, (45, 160), (120, 185), thickness=2, jitter=1)
  card_path = os.path.join(output_dir, "card_sketch.png")
  cv2.imwrite(card_path, card_canvas)

  # 3. InputField Sketch
  input_canvas = np.ones((227, 227, 3), dtype=np.uint8) * 248
  input_canvas = np.clip(input_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
  add_sketchy_rect(input_canvas, (25, 95), (200, 135), thickness=3, jitter=2)
  cv2.putText(input_canvas, "Email...", (40, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)
  input_path = os.path.join(output_dir, "input_sketch.png")
  cv2.imwrite(input_path, input_canvas)

  # 4. Navbar Sketch
  nav_canvas = np.ones((227, 227, 3), dtype=np.uint8) * 248
  nav_canvas = np.clip(nav_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
  add_sketchy_rect(nav_canvas, (10, 15), (217, 65), thickness=3, jitter=2)
  # Logo square
  add_sketchy_rect(nav_canvas, (20, 25), (55, 55), thickness=2, jitter=1)
  # Nav links
  add_sketchy_line(nav_canvas, (80, 40), (115, 40), thickness=2, jitter=1)
  add_sketchy_line(nav_canvas, (130, 40), (165, 40), thickness=2, jitter=1)
  # Avatar circle
  cv2.circle(nav_canvas, (195, 40), 12, (40, 40, 40), 2)
  nav_path = os.path.join(output_dir, "navbar_sketch.png")
  cv2.imwrite(nav_path, nav_canvas)

  # 5. Modal Sketch
  modal_canvas = np.ones((227, 227, 3), dtype=np.uint8) * 248
  modal_canvas = np.clip(modal_canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
  add_sketchy_rect(modal_canvas, (40, 35), (185, 190), thickness=3, jitter=2)
  # Header line
  add_sketchy_line(modal_canvas, (40, 70), (185, 70), thickness=2, jitter=1)
  # Close 'X'
  add_sketchy_line(modal_canvas, (165, 45), (175, 55), thickness=2, jitter=1)
  add_sketchy_line(modal_canvas, (175, 45), (165, 55), thickness=2, jitter=1)
  # Body lines
  add_sketchy_line(modal_canvas, (55, 95), (170, 95), thickness=1, jitter=1)
  add_sketchy_line(modal_canvas, (55, 115), (150, 115), thickness=1, jitter=1)
  # Action buttons (Cancel & Confirm)
  add_sketchy_rect(modal_canvas, (55, 145), (105, 175), thickness=2, jitter=1)
  add_sketchy_rect(modal_canvas, (120, 145), (170, 175), thickness=2, jitter=1)
  modal_path = os.path.join(output_dir, "modal_sketch.png")
  cv2.imwrite(modal_path, modal_canvas)

  print(f"[Generated] 5 realistic UI sketches saved to '{output_dir}/'")
  return [btn_path, card_path, input_path, nav_path, modal_path]


if __name__ == "__main__":
  create_sample_sketches()
