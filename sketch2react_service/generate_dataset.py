"""Generates a labelled synthetic dataset of hand-drawn UI wireframe sketches.

Every sample is randomized (size, position, pen thickness, pen darkness, stroke jitter,
rotation, paper noise, inner details) so the AlexNet backbone learns the *shape* of each
component instead of memorizing one fixed drawing.

Output layout (ImageFolder style, readable by keras.utils.image_dataset_from_directory):

    dataset/
      train/{Button,Card,InputField,Navbar,Modal}/00000.png
      val/...
      test/...

Usage:
    python generate_dataset.py                      # 800 / 150 / 150 per class
    python generate_dataset.py --train 300 --val 50 --test 50 --out dataset_small
"""

import argparse
import os
import random

import cv2
import numpy as np

UI_CLASSES = ["Button", "Card", "InputField", "Navbar", "Modal"]
IMG_SIZE = 227

WORDS = ["Submit", "Sign Up", "Login", "OK", "Buy", "Next", "Save", "Go", "Send", "Join", "Start", "More"]
PLACEHOLDERS = ["Email...", "Search", "Name", "Password", "type here", "Username", "Enter text", "Phone"]
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]


class Pen:
  """Random pen style shared by all strokes of one sketch (pencil, ballpoint, marker, canvas brush)."""

  def __init__(self, rng: random.Random):
    if rng.random() < 0.15:
      gray = rng.randint(90, 150)  # light pencil
    else:
      gray = rng.randint(5, 70)  # pen / marker / canvas stroke (#0f172a)
    self.color = (gray, gray, gray)
    self.thickness = rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5])
    self.jitter = rng.choice([0, 0, 1, 1, 2, 2, 3])
    self.rng = rng

  def line(self, canvas, p1, p2, thickness=None):
    """Wobbly hand-drawn line between two points."""
    t = thickness or self.thickness
    (x1, y1), (x2, y2) = p1, p2
    segments = max(2, int(np.hypot(x2 - x1, y2 - y1) / 15))
    xs = np.linspace(x1, x2, segments + 1)
    ys = np.linspace(y1, y2, segments + 1)
    j = self.jitter
    pts = [(int(x + self.rng.randint(-j, j)), int(y + self.rng.randint(-j, j))) for x, y in zip(xs, ys)]
    for a, b in zip(pts[:-1], pts[1:]):
      cv2.line(canvas, a, b, self.color, t, cv2.LINE_AA)

  def rect(self, canvas, x1, y1, x2, y2, thickness=None):
    # Hand-drawn rectangles often overshoot their corners slightly
    o = self.rng.randint(0, 3)
    self.line(canvas, (x1 - o, y1), (x2 + o, y1), thickness)
    self.line(canvas, (x2, y1 - o), (x2, y2 + o), thickness)
    self.line(canvas, (x2 + o, y2), (x1 - o, y2), thickness)
    self.line(canvas, (x1, y2 + o), (x1, y1 - o), thickness)

  def rounded_rect(self, canvas, x1, y1, x2, y2, r):
    r = max(2, min(r, (x2 - x1) // 3, (y2 - y1) // 3))
    self.line(canvas, (x1 + r, y1), (x2 - r, y1))
    self.line(canvas, (x2, y1 + r), (x2, y2 - r))
    self.line(canvas, (x2 - r, y2), (x1 + r, y2))
    self.line(canvas, (x1, y2 - r), (x1, y1 + r))
    for cx, cy, a in [(x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270), (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90)]:
      cv2.ellipse(canvas, (cx, cy), (r, r), a, 0, 90, self.color, self.thickness, cv2.LINE_AA)

  def box(self, canvas, x1, y1, x2, y2):
    """Rectangle that is randomly sharp or rounded."""
    if self.rng.random() < 0.35:
      self.rounded_rect(canvas, x1, y1, x2, y2, self.rng.randint(4, 14))
    else:
      self.rect(canvas, x1, y1, x2, y2)

  def circle(self, canvas, c, r, thickness=None):
    cv2.circle(canvas, (int(c[0]), int(c[1])), int(r), self.color, thickness or self.thickness, cv2.LINE_AA)

  def text_or_squiggle(self, canvas, x, y, max_w, words, light=False):
    """Writes a short word, or scribbles a wavy line the way people fake text in wireframes."""
    color = tuple(min(200, c + 70) for c in self.color) if light else self.color
    if self.rng.random() < 0.5:
      word = self.rng.choice(words)
      font = self.rng.choice(FONTS)
      scale = 0.35
      while scale < 0.9 and cv2.getTextSize(word, font, scale + 0.05, 1)[0][0] < max_w:
        scale += 0.05
      scale = self.rng.uniform(0.35, scale)
      cv2.putText(canvas, word, (int(x), int(y)), font, scale, color, self.rng.choice([1, 1, 2]), cv2.LINE_AA)
    else:
      w = int(max_w * self.rng.uniform(0.4, 1.0))
      xs = np.arange(int(x), int(x) + max(6, w), 3)
      ys = y - 4 + 3 * np.sin(xs / self.rng.uniform(2.0, 4.0))
      pts = np.stack([xs, ys], axis=1).astype(np.int32)
      cv2.polylines(canvas, [pts], False, color, max(1, self.thickness - 1), cv2.LINE_AA)

  def text_lines(self, canvas, x1, y, x2, n, gap):
    for i in range(n):
      end = x2 if i < n - 1 else x1 + int((x2 - x1) * self.rng.uniform(0.4, 0.9))
      self.line(canvas, (x1, y + i * gap), (end, y + i * gap), max(1, self.thickness - 1))


# ---------------------------------------------------------------------------
# One drawing function per UI class
# ---------------------------------------------------------------------------
def draw_button(canvas, pen, rng):
  w = rng.randint(60, 180)
  h = int(np.clip(w / rng.uniform(1.8, 4.5), 22, 70))
  x1 = rng.randint(8, IMG_SIZE - w - 8)
  y1 = rng.randint(30, IMG_SIZE - h - 15)
  x2, y2 = x1 + w, y1 + h
  if rng.random() < 0.3:
    pen.rounded_rect(canvas, x1, y1, x2, y2, h // 2)  # pill button
  else:
    pen.box(canvas, x1, y1, x2, y2)
  if rng.random() < 0.15:  # filled / shaded primary button
    for yy in range(y1 + 4, y2 - 2, rng.randint(3, 6)):
      pen.line(canvas, (x1 + 4, yy), (x2 - 4, yy), 1)
  if rng.random() < 0.85:  # centered label
    tw = int(w * 0.55)
    pen.text_or_squiggle(canvas, x1 + (w - tw) // 2, y1 + h // 2 + 5, tw, WORDS)
  if rng.random() < 0.15:  # icon inside button
    pen.circle(canvas, (x1 + 12, y1 + h // 2), max(3, h // 6), 1)


def draw_input(canvas, pen, rng):
  w = rng.randint(120, 215)
  h = int(np.clip(w / rng.uniform(4.0, 8.0), 18, 45))
  x1 = rng.randint(5, IMG_SIZE - w - 5)
  y1 = rng.randint(35, IMG_SIZE - h - 15)
  x2, y2 = x1 + w, y1 + h
  if rng.random() < 0.12:
    pen.line(canvas, (x1, y2), (x2, y2))  # underline-only material input
  else:
    pen.box(canvas, x1, y1, x2, y2)
  if rng.random() < 0.8:  # left-aligned light placeholder
    pen.text_or_squiggle(canvas, x1 + 8, y1 + h // 2 + 5, int(w * 0.5), PLACEHOLDERS, light=True)
  if rng.random() < 0.4:  # label above the field
    pen.text_or_squiggle(canvas, x1, y1 - 7, int(w * 0.35), ["Email", "Name", "Password", "Search"])
  if rng.random() < 0.25:  # caret or search icon
    if rng.random() < 0.5:
      pen.line(canvas, (x2 - 15, y1 + 5), (x2 - 15, y2 - 5), 1)
    else:
      pen.circle(canvas, (x2 - 14, y1 + h // 2), max(3, h // 5), 1)


def draw_card(canvas, pen, rng):
  w = rng.randint(100, 200)
  h = rng.randint(max(110, int(w * 0.9)), 215)
  x1 = rng.randint(5, IMG_SIZE - w - 5)
  y1 = rng.randint(5, IMG_SIZE - h - 5)
  x2, y2 = x1 + w, y1 + h
  pen.box(canvas, x1, y1, x2, y2)
  pad = rng.randint(8, 15)
  y = y1 + pad
  if rng.random() < 0.85:  # image placeholder with a cross
    ih = int(h * rng.uniform(0.3, 0.45))
    pen.rect(canvas, x1 + pad, y, x2 - pad, y + ih, max(1, pen.thickness - 1))
    if rng.random() < 0.8:
      pen.line(canvas, (x1 + pad, y), (x2 - pad, y + ih), 1)
      pen.line(canvas, (x2 - pad, y), (x1 + pad, y + ih), 1)
    else:  # mountain / sun icon
      pen.circle(canvas, (x1 + pad + 15, y + 12), 5, 1)
    y += ih + rng.randint(10, 16)
  n_lines = rng.randint(1, 3)
  pen.text_lines(canvas, x1 + pad, y, x2 - pad, n_lines, rng.randint(10, 15))
  y += n_lines * 13 + 8
  if rng.random() < 0.6 and y2 - y > 25:  # single CTA button at bottom
    bh = rng.randint(16, min(30, y2 - y - 6))
    bw = int((x2 - x1 - 2 * pad) * rng.uniform(0.4, 1.0))
    by = y2 - pad - bh
    pen.rect(canvas, x1 + pad, by, x1 + pad + bw, by + bh, max(1, pen.thickness - 1))


def draw_navbar(canvas, pen, rng):
  x1 = rng.randint(0, 15)
  x2 = IMG_SIZE - rng.randint(0, 15)
  y1 = rng.randint(0, 25)
  h = rng.randint(28, 60)
  y2 = y1 + h
  if rng.random() < 0.2:
    pen.line(canvas, (x1, y2), (x2, y2))  # only a bottom border
  else:
    pen.box(canvas, x1, y1, x2, y2)
  cy = (y1 + y2) // 2
  lx = x1 + rng.randint(8, 15)
  s = max(8, int(h * 0.55))
  if rng.random() < 0.5:  # logo
    pen.rect(canvas, lx, cy - s // 2, lx + s, cy + s // 2, max(1, pen.thickness - 1))
  else:
    pen.circle(canvas, (lx + s // 2, cy), s // 2, max(1, pen.thickness - 1))
  if rng.random() < 0.5:
    pen.text_or_squiggle(canvas, lx + s + 5, cy + 5, 40, ["Brand", "Logo", "App"])
  n_links = rng.randint(2, 4)
  link_start = x1 + int((x2 - x1) * rng.uniform(0.35, 0.5))
  right_icon = rng.random() < 0.75
  link_end = x2 - (40 if right_icon else 10)
  step = (link_end - link_start) / n_links
  for i in range(n_links):
    a = int(link_start + i * step)
    pen.line(canvas, (a, cy), (a + int(step * 0.65), cy), max(1, pen.thickness - 1))
  if right_icon:
    rx = x2 - 22
    if rng.random() < 0.5:  # avatar
      pen.circle(canvas, (rx, cy), max(6, h // 4), max(1, pen.thickness - 1))
    else:  # hamburger
      for dy in (-6, 0, 6):
        pen.line(canvas, (rx - 9, cy + dy), (rx + 9, cy + dy), max(1, pen.thickness - 1))
  if rng.random() < 0.2:  # faint page content below the bar
    pen.text_lines(canvas, x1 + 20, y2 + rng.randint(30, 80), x2 - 40, rng.randint(1, 3), 14)


def draw_modal(canvas, pen, rng):
  w = rng.randint(115, 190)
  h = rng.randint(95, 185)
  cx = IMG_SIZE // 2 + rng.randint(-15, 15)
  cy = IMG_SIZE // 2 + rng.randint(-15, 15)
  x1, y1 = max(4, cx - w // 2), max(4, cy - h // 2)
  x2, y2 = min(IMG_SIZE - 4, x1 + w), min(IMG_SIZE - 4, y1 + h)
  if rng.random() < 0.25:  # dimmed backdrop hatching outside the dialog
    for d in range(-IMG_SIZE, IMG_SIZE, rng.randint(14, 22)):
      cv2.line(canvas, (d, 0), (d + IMG_SIZE, IMG_SIZE), tuple(min(220, c + 120) for c in pen.color), 1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (250, 250, 250), -1)
  pen.box(canvas, x1, y1, x2, y2)
  pad = rng.randint(8, 14)
  header_y = y1 + rng.randint(22, 35)
  if rng.random() < 0.75:  # header divider
    pen.line(canvas, (x1, header_y), (x2, header_y), max(1, pen.thickness - 1))
  pen.text_or_squiggle(canvas, x1 + pad, header_y - 8, int((x2 - x1) * 0.45), ["Confirm", "Delete?", "Title", "Alert", "Warning"])
  if rng.random() < 0.85:  # close X in top-right corner
    s = rng.randint(5, 8)
    xx, yy = x2 - pad - s, y1 + (header_y - y1) // 2
    pen.line(canvas, (xx - s, yy - s), (xx + s, yy + s), max(1, pen.thickness - 1))
    pen.line(canvas, (xx + s, yy - s), (xx - s, yy + s), max(1, pen.thickness - 1))
  bh = rng.randint(16, 26)
  by2 = y2 - pad
  by1 = by2 - bh
  body_space = by1 - header_y - 10
  if body_space > 14:
    pen.text_lines(canvas, x1 + pad, header_y + 14, x2 - pad, min(3, body_space // 14), 13)
  n_buttons = 2 if rng.random() < 0.75 else 1
  bw = rng.randint(32, max(33, min(60, (x2 - x1 - 3 * pad) // n_buttons)))
  bx2 = x2 - pad
  for _ in range(n_buttons):
    pen.rect(canvas, bx2 - bw, by1, bx2, by2, max(1, pen.thickness - 1))
    bx2 -= bw + rng.randint(6, 12)


DRAWERS = {
    "Button": draw_button,
    "Card": draw_card,
    "InputField": draw_input,
    "Navbar": draw_navbar,
    "Modal": draw_modal,
}


def make_paper(rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
  """White canvas (like the /demo page) or slightly off-white noisy paper (like a photo)."""
  if rng.random() < 0.4:
    return np.full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=np.uint8)
  base = rng.randint(225, 252)
  noise = np_rng.normal(0, rng.uniform(1, 5), (IMG_SIZE, IMG_SIZE, 1))
  return np.clip(base + noise, 0, 255).repeat(3, axis=2).astype(np.uint8)


def render_sketch(label: str, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
  canvas = make_paper(rng, np_rng)
  bg = int(canvas[0, 0, 0])
  pen = Pen(rng)
  DRAWERS[label](canvas, pen, rng)

  # Small rotation + shift: nobody draws perfectly axis-aligned
  angle = rng.uniform(-4, 4)
  M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, 1.0)
  M[:, 2] += [rng.randint(-6, 6), rng.randint(-6, 6)]
  canvas = cv2.warpAffine(canvas, M, (IMG_SIZE, IMG_SIZE), borderValue=(bg, bg, bg))
  if rng.random() < 0.2:
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)  # phone-camera softness
  return canvas


def generate(out_dir: str, splits: dict, seed: int = 7):
  rng = random.Random(seed)
  np_rng = np.random.default_rng(seed)
  for split, per_class in splits.items():
    for label in UI_CLASSES:
      class_dir = os.path.join(out_dir, split, label)
      os.makedirs(class_dir, exist_ok=True)
      for i in range(per_class):
        img = render_sketch(label, rng, np_rng)
        cv2.imwrite(os.path.join(class_dir, f"{i:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    print(f"[Dataset] {split:<5}: {per_class} x {len(UI_CLASSES)} classes = {per_class * len(UI_CLASSES)} images")
  print(f"[Dataset] Saved to '{out_dir}/'")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--out", default="dataset")
  parser.add_argument("--train", type=int, default=800, help="images per class")
  parser.add_argument("--val", type=int, default=150, help="images per class")
  parser.add_argument("--test", type=int, default=150, help="images per class")
  parser.add_argument("--seed", type=int, default=7)
  args = parser.parse_args()
  generate(args.out, {"train": args.train, "val": args.val, "test": args.test}, seed=args.seed)
