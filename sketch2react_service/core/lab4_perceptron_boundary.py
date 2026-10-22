"""Lab 4: The Perceptron & Logic Gates / Hyperplane Boundary Analysis.

Used in Sketch2React for:
- Fast geometric layout classification (e.g. Button vs Card based on Aspect Ratio and Bounding Box Area)
- Evaluating linear separability and computing exact decision thresholds (-w0) without iterative training
"""

from typing import List, Optional, Tuple


def perceptron(x1: float, x2: float, w1: float, w2: float, w0: float) -> int:
  """Single Perceptron activation: 1 if (w1*x1 + w2*x2 + w0) >= 0 else 0."""
  return 1 if (w1 * x1 + w2 * x2 + w0) >= 0 else 0


def get_minus_w0(w1: float, w2: float, gate_name: str, truth_table: List[int]) -> Optional[float]:
  """Lab 4 Part B: For given (w1, w2), outputs a valid -w0 such that the truth table is satisfied.

  Condition:
      w1*x1 + w2*x2 + w0 >= 0  <=>  w1*x1 + w2*x2 >= -w0
  Therefore:
      For y = 1: -w0 <= w1*x1 + w2*x2  (Upper bound on -w0)
      For y = 0: -w0 >  w1*x1 + w2*x2  (Lower bound on -w0)
  """
  inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
  lower_bound = -float("inf")
  upper_bound = float("inf")

  for (x1, x2), y in zip(inputs, truth_table):
    val = w1 * x1 + w2 * x2
    if y == 1:
      upper_bound = min(upper_bound, val)
    else:
      lower_bound = max(lower_bound, val)

  if lower_bound < upper_bound:
    # Any value in (lower_bound, upper_bound] works; picking upper_bound if finite
    minus_w0 = upper_bound if upper_bound != float("inf") else lower_bound + 1.0
    return float(minus_w0)
  else:
    return None  # Linearly inseparable gate (e.g. XOR)


def classify_button_vs_card(aspect_ratio: float, normalized_area: float) -> str:
  """Sketch2React Layout Gate: Uses a calibrated Perceptron hyperplane to distinguish

  a Button (elongated, small area) from a Card (square/tall, large area).
  """
  # Using calibrated weights: w1 = 2.0 (aspect ratio), w2 = -3.0 (area), w0 = -0.5
  score = 2.0 * aspect_ratio - 3.0 * normalized_area - 0.5
  return "Button" if score >= 0 else "Card"


# --- Interactive Lab Revision Demo ---
if __name__ == "__main__":
  print("=" * 60)
  print("LAB 4: PERCEPTRON & LOGIC GATES (-w0 DERIVATION)")
  print("=" * 60)

  truth_tables = {
      "AND": [0, 0, 0, 1],
      "OR": [0, 1, 1, 1],
      "NOR": [1, 0, 0, 0],
  }

  print("\n1. Finding valid -w0 given weights:")
  for gate, table in truth_tables.items():
    if gate == "NOR":
      w1, w2 = -1.0, -1.0
    else:
      w1, w2 = 1.0, 1.0
    minus_w0 = get_minus_w0(w1, w2, gate, table)
    print(f"Gate {gate:<4} | Weights: ({w1}, {w2}) -> Valid -w0: {minus_w0} (Bias w0 = {-minus_w0})")

  print("\n2. Sketch Layout Decision Boundary Check:")
  test_boxes = [
      ("Elongated small box (3.2 aspect ratio, 0.15 area)", 3.2, 0.15),
      ("Large square box    (1.1 aspect ratio, 0.65 area)", 1.1, 0.65)
  ]
  for desc, ar, area in test_boxes:
    comp = classify_button_vs_card(ar, area)
    print(f"Box: {desc} -> Classified as: {comp}")
  print("=" * 60)
