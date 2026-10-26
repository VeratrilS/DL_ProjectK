"""Sketch2React Master Pipeline.

Unifies the Deep Learning Labs + Fraz's DSA Algorithmic Suite:
- Lab 0: Orthogonal rotation matrix used to deskew slanted sketches
- Lab 1: TensorFlow tensor arithmetic
- Lab 2: First-principles scratch MLP forward pass (informational)
- Lab 3: Batch Normalization of pixel intensities for pen-contrast invariance
- Lab 4: Perceptron hyperplane layout gate (Button vs Card)
- Lab 5: Multi-layer ANN with Dropout (style head)
- Lab 7: AlexNet with Custom Local Response Normalization (LRN) -- the trained classifier
- Fraz's DSA Suite: 2D Prefix Sum, Two Pointers, Monotonic Stack, Merge Intervals, BFS, TopoSort, Trie, Heaps, DP

Train the classifier first (`python generate_dataset.py && python train.py`). Without trained
weights the pipeline falls back to hand-written geometric rules (~74% test accuracy).
"""

import base64
import io
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from core.fraz_algorithms import (
    FLAG_BORDER,
    FLAG_HOVER,
    FLAG_ROUNDED,
    FLAG_SHADOW,
    PrefixSum2D,
    ReactComponentTrie,
    TreeNode,
    binary_search_lower_bound,
    binary_search_viewport_capacity,
    container_with_most_water,
    dijkstra_shortest_path,
    evaluate_bitmask_flags,
    find_stroke_islands_bfs,
    kahn_topological_sort_jsx,
    knapsack_viewport_packing,
    largest_rectangle_in_histogram,
    layout_edit_distance,
    lowest_common_ancestor_dom,
    merge_bounding_intervals,
    sliding_window_maximum,
    subarray_sum_equals_k,
    top_k_confident_detections,
    trapping_rain_water,
    two_pointers_tight_bounds,
    validate_jsx_nesting_tags,
)
from core.lab0_linear_algebra import (
    compute_trace,
    create_orthogonal_rotation_matrix,
    determinant_via_gaussian_elimination,
)
from core.lab1_tf_graph import tensor_arithmetic_demo
from core.lab2_scratch_backprop import ScratchMLP
from core.lab4_perceptron_boundary import classify_button_vs_card, get_minus_w0
from core.lab5_style_ann import build_lab5_style_ann, predict_component_style
from core.lab7_alexnet_backbone import (
    UI_CLASSES,
    load_trained_alexnet,
    preprocess_sketch_for_alexnet,
    to_grayscale_227,
)

IMG_SIZE = 227
INK_THRESHOLD = 0.2  # a pixel counts as ink when (1 - luminance) > 0.2
MIN_INK_PIXELS = 30
MAX_DESKEW_DEGREES = 10.0

REACT_TEMPLATES = {
    "Button": """export default function ButtonComponent() {
  return (
    <button className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-xl shadow-md transition-all duration-200 hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-blue-500/50">
      Submit Action
    </button>
  );
}""",
    "Card": """export default function CardComponent() {
  return (
    <div className="max-w-sm rounded-2xl border border-slate-700 bg-slate-900/80 p-6 shadow-xl backdrop-blur-md">
      <div className="h-36 w-full rounded-xl bg-slate-800 flex items-center justify-center text-slate-500 font-medium">
        Image Placeholder
      </div>
      <h3 className="mt-4 text-xl font-semibold text-white">Modern Card Title</h3>
      <p className="mt-2 text-sm text-slate-400">Hand-drawn wireframe parsed into responsive Tailwind component.</p>
      <button className="mt-5 w-full rounded-xl bg-blue-600 py-2.5 font-medium text-white shadow-lg transition hover:bg-blue-700">
        Explore More
      </button>
    </div>
  );
}""",
    "InputField": """export default function InputFieldComponent() {
  return (
    <div className="w-full max-w-md">
      <label htmlFor="email" className="block text-sm font-medium text-slate-300 mb-1.5">Email Address</label>
      <input
        id="email"
        type="email"
        placeholder="Enter your email..."
        className="w-full px-4 py-2.5 rounded-xl border border-slate-700 bg-slate-900 text-white placeholder-slate-500 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all outline-none"
      />
    </div>
  );
}""",
    "Navbar": """export default function NavbarComponent() {
  return (
    <nav className="w-full flex items-center justify-between px-6 py-3.5 bg-slate-900/90 border-b border-slate-800 backdrop-blur-md">
      <div className="flex items-center gap-2 text-blue-400 font-bold text-lg">
        <span className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center text-white">⚡</span>
        <span>BrandApp</span>
      </div>
      <div className="flex items-center gap-6 text-sm text-slate-300">
        <a href="#features" className="hover:text-white transition">Features</a>
        <a href="#pricing" className="hover:text-white transition">Pricing</a>
        <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg shadow-sm">
          Get Started
        </button>
      </div>
    </nav>
  );
}""",
    "Modal": """export default function ModalComponent({ isOpen = true, onClose }) {
  if (!isOpen) return null;
  return (
    <div role="dialog" aria-modal="true" className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-2xl">
        <h3 className="text-xl font-bold text-white">Confirm Dialog</h3>
        <p className="mt-2 text-sm text-slate-400">Are you sure you want to proceed with this action?</p>
        <div className="mt-6 flex justify-end gap-3">
          <button onClick={onClose} className="px-4 py-2 rounded-xl border border-slate-700 text-slate-300 hover:bg-slate-800 transition text-sm font-medium">
            Cancel
          </button>
          <button className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white transition text-sm font-medium shadow-md">
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}""",
}


def load_image(image_input) -> np.ndarray:
  """Accepts a file path, a base64 data URL, or a numpy array and returns an RGB/gray uint8 array."""
  if isinstance(image_input, np.ndarray):
    return image_input
  if not isinstance(image_input, str):
    raise TypeError("Unsupported image input type")
  if image_input.startswith("data:image"):
    encoded = image_input.split(",", 1)[1]
    return np.array(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))
  img = cv2.imread(image_input)
  if img is None:
    raise ValueError(f"Could not load image from {image_input}")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def deskew(gray: np.ndarray, ink_mask: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
  """Lab 0: estimates the sketch's slant from the ink's minimum-area rectangle and rotates it
  back with an orthogonal rotation matrix R (R . R^T = I, det(R) = 1)."""
  ys, xs = np.nonzero(ink_mask)
  angle = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))[2]
  # OpenCV versions disagree on the range ([-90, 0) vs (0, 90]); a rectangle's slant is defined mod 90
  angle = ((angle + 45) % 90) - 45
  if abs(angle) > MAX_DESKEW_DEGREES:
    angle = 0.0  # large angles are more likely intentional shapes than slant
  R = create_orthogonal_rotation_matrix(-angle)
  center = np.array([IMG_SIZE / 2, IMG_SIZE / 2])
  affine = np.hstack([R, (center - R @ center).reshape(2, 1)])
  if angle:
    border = int(np.median(gray))
    gray = cv2.warpAffine(gray, affine, (IMG_SIZE, IMG_SIZE), borderValue=border)
  return gray, angle, R


def rule_based_classify(geom: dict, layout_gate_decision: str) -> Tuple[str, float]:
  """Fallback classifier used only when no trained AlexNet weights are available."""
  t, w, h = geom["top"], geom["width"], geom["height"]
  ar, area, y_center = geom["aspect_ratio"], geom["normalized_area"], geom["y_center"]
  if t <= 35 and y_center <= 0.35 and w >= 155:
    return "Navbar", 0.942
  if area >= 0.55 and h >= 160:
    return "Card", 0.926
  if 0.35 <= y_center <= 0.65 and 0.75 <= ar <= 1.35 and 0.25 <= area < 0.55:
    return "Modal", 0.895
  if ar >= 3.2 or (h <= 55 and w >= 155 and y_center > 0.35):
    return "InputField", 0.914
  return layout_gate_decision, 0.931


def style_variant_for(component: str, ink_density: float, aspect_ratio: float) -> str:
  if component == "Modal":
    return "Glassmorphism / Shadow"
  if component == "Card":
    return "Elevated Modern Card"
  if ink_density > 0.09:
    return "Primary Filled"
  if aspect_ratio > 3.0:
    return "Outline Bordered"
  return "Subtle Minimal"


class Sketch2ReactPipeline:
  """End-to-End Multimodal Vision-to-React Component Pipeline."""

  def __init__(self):
    print("\n[Pipeline] Initializing Sketch2React Master Pipeline...")

    # Trie for React/Tailwind autocomplete (Fraz DSA)
    self.trie = ReactComponentTrie()
    for tag in [
        "Button", "Card", "CardHeader", "CardContent", "Input", "Navbar", "Modal", "Dialog",
        "flex", "grid", "rounded-xl", "shadow-lg", "p-4", "hover:bg-blue-600"
    ]:
      self.trie.insert(tag)

    self.scratch_calibrator = ScratchMLP(n_in=3, n_h1=3, n_h2=3, n_out=2)  # Lab 2
    self.style_ann = build_lab5_style_ann(input_shape=(28, 28), num_classes=5)  # Lab 5

    # Lab 7: AlexNet trained by train.py
    self.alexnet = load_trained_alexnet()
    if self.alexnet is None:
      print("[Pipeline] WARNING: no trained AlexNet weights found (run train.py). Using geometric rule fallback.")
    else:
      print("[Pipeline] Loaded trained AlexNet weights.")

  def preprocess_sketch_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Stage 1: grayscale + Lab 0 deskew + geometry via Fraz 2D Prefix Sum / Two Pointers.

    Returns (gray_227, ink_map, geometry) where ink_map is 1 on strokes and 0 on paper.
    """
    gray = to_grayscale_227(img)
    ink_mask = (1.0 - gray / 255.0) > INK_THRESHOLD
    total_ink = int(ink_mask.sum())
    geom = {"total_ink_pixels": total_ink, "deskew_angle": 0.0, "rotation_trace": 2.0, "rotation_det": 1.0}
    if total_ink < MIN_INK_PIXELS:
      return gray, ink_mask.astype(np.float32), geom

    gray, angle, R = deskew(gray, ink_mask)
    ink = 1.0 - gray.astype(np.float32) / 255.0
    ink_mask = ink > INK_THRESHOLD

    # Fraz 2D Prefix Sum: O(1) ink density of the whole canvas
    density = PrefixSum2D(ink.tolist()).query_box_density(0, 0, IMG_SIZE - 1, IMG_SIZE - 1)

    # Fraz Two Pointers on column / row ink projections -> tight bounding box
    left, right = two_pointers_tight_bounds(ink_mask.mean(axis=0).tolist(), threshold=0.02)
    top, bottom = two_pointers_tight_bounds(ink_mask.mean(axis=1).tolist(), threshold=0.02)
    width, height = max(10, right - left), max(10, bottom - top)

    geom.update({
        "deskew_angle": round(float(angle), 2),
        "rotation_trace": compute_trace(R),
        "rotation_det": determinant_via_gaussian_elimination(R.copy()),
        "ink_density": float(density),
        "horizontal_bounds": [int(left), int(right)],
        "vertical_bounds": [int(top), int(bottom)],
        "top": int(top),
        "width": int(width),
        "height": int(height),
        "aspect_ratio": width / height,
        "normalized_area": (width * height) / (IMG_SIZE * IMG_SIZE),
        "y_center": (top + bottom) / (2.0 * IMG_SIZE),
    })
    return gray, ink, geom

  def classify(self, gray: np.ndarray, geom: dict, layout_gate_decision: str) -> Tuple[str, Dict[str, float], str]:
    """Returns (component, per-class probabilities, source)."""
    if self.alexnet is not None:
      x = preprocess_sketch_for_alexnet(gray)[np.newaxis]
      probs = self.alexnet(x, training=False).numpy()[0]
      return UI_CLASSES[int(np.argmax(probs))], {c: float(p) for c, p in zip(UI_CLASSES, probs)}, "alexnet"

    component, conf = rule_based_classify(geom, layout_gate_decision)
    rest = (1.0 - conf) / (len(UI_CLASSES) - 1)
    return component, {c: conf if c == component else rest for c in UI_CLASSES}, "rules"

  def process(self, image_input) -> dict:
    """Executes the complete end-to-end pipeline on an uploaded sketch.

    Accepts: file path, base64 data URL, or numpy array.
    """
    img = load_image(image_input)
    start_time = time.time()

    gray, ink, geom = self.preprocess_sketch_image(img)
    if geom["total_ink_pixels"] < MIN_INK_PIXELS:
      return self._empty_result(start_time)

    # Connected stroke components on a 28x28 ink patch (Fraz BFS / LeetCode 200)
    patch_28 = cv2.resize(ink, (28, 28), interpolation=cv2.INTER_AREA)
    binary_grid = (patch_28 > 0.1).astype(int).tolist()
    stroke_islands = find_stroke_islands_bfs(binary_grid)

    # Lab 4 Perceptron gate + Lab 7 AlexNet classification
    layout_gate_decision = classify_button_vs_card(geom["aspect_ratio"], geom["normalized_area"])
    minus_w0_val = get_minus_w0(1.0, 1.0, "AND", [0, 0, 0, 1])
    component, probs, source = self.classify(gray, geom, layout_gate_decision)
    confidence = probs[component]

    # Heap top-k + binary search lower bound over the class probabilities
    top_candidates = top_k_confident_detections(list(probs.items()), k=3)
    cutoff_rank = binary_search_lower_bound(sorted(probs.values()), min_confidence=0.05)

    # Lab 5 style head (untrained: there are no style labels yet) + geometric style rules
    style_ann_out = predict_component_style(self.style_ann, patch_28)
    style_variant = style_variant_for(component, geom["ink_density"], geom["aspect_ratio"])

    # Lab 2 scratch MLP forward pass on the geometric features
    _, _, calib_out = self.scratch_calibrator.forward([
        min(1.0, geom["aspect_ratio"] / 4.0), min(1.0, geom["normalized_area"]), confidence
    ])

    # Lab 1 tensor arithmetic
    tensor_add, tensor_sub = tensor_arithmetic_demo(confidence, 0.05)

    # Column ink histogram features (Monotonic Stack, Two Pointers, Deque, Prefix Sum)
    col_heights = [int(v) for v in np.sum(np.array(binary_grid), axis=0)]
    max_area, left_col, right_col, max_h = largest_rectangle_in_histogram(col_heights)
    trapped_water = trapping_rain_water(col_heights)
    max_water, _, _ = container_with_most_water(col_heights)
    smoothed_peaks = sliding_window_maximum(col_heights, k=3)
    divider_count = subarray_sum_equals_k(col_heights, k=5)

    # NMS: merge the horizontal extents of the stroke islands (LeetCode 56)
    scale = IMG_SIZE / 28.0
    island_intervals = [[int(min(c for _, c in isl) * scale), int((max(c for _, c in isl) + 1) * scale)] for isl in stroke_islands]
    merged_intervals = merge_bounding_intervals(island_intervals, tolerance=int(scale))

    # JSX tree: Kahn TopoSort, tag matcher, Dijkstra, LCA, edit distance
    tree_nodes = ["PageWrapper", "Container", component]
    jsx_hierarchy = kahn_topological_sort_jsx(tree_nodes, [("PageWrapper", "Container"), ("Container", component)])
    is_jsx_valid = validate_jsx_nesting_tags([f"<{t}>" for t in jsx_hierarchy] + [f"</{t}>" for t in reversed(jsx_hierarchy)])
    route_dist = dijkstra_shortest_path(tree_nodes, [("PageWrapper", "Container", 1), ("Container", component, 2)], "PageWrapper")
    root = TreeNode("PageWrapper")
    root.left = TreeNode("Container")
    root.left.left = TreeNode(component)
    lca = lowest_common_ancestor_dom(root, "Container", component)
    template_edit_dist = layout_edit_distance(jsx_hierarchy, ["PageWrapper", "Container", "Card", "Button"])

    # Viewport DP / binary search, CSS bitmask flags, Trie autocomplete
    viewport_cap = binary_search_viewport_capacity([geom["height"], 80, 120], max_pages=2)
    packed_priority = knapsack_viewport_packing([80, 120, 200], [15, 30, 45], max_capacity=IMG_SIZE)
    css_features = evaluate_bitmask_flags(FLAG_BORDER | FLAG_SHADOW | FLAG_ROUNDED | FLAG_HOVER)
    tailwind_classes = self.trie.autocomplete("rounded") + self.trie.autocomplete("hover:")

    suggested_prompt = (
        f"Generate a pixel-perfect, responsive React component for a '{component}' "
        f"with '{style_variant}' design style. "
        f"Use Tailwind CSS ({', '.join(css_features)}) with clean micro-interactions, responsive flex layout, "
        f"and accessible ARIA labels. Nested hierarchy: {' > '.join(jsx_hierarchy)}."
    )
    elapsed_ms = (time.time() - start_time) * 1000.0
    classifier_desc = "Trained AlexNet (Custom LRN)" if source == "alexnet" else "Geometric rules (AlexNet not trained yet)"

    return {
        "status": "success",
        "detected_component": component,
        "confidence": f"{confidence * 100:.2f}%",
        "classifier": source,
        "style_variant": style_variant,
        "top_3_candidates": [{"component": c, "confidence": f"{s * 100:.2f}%"} for c, s in top_candidates],
        "generated_react_code": REACT_TEMPLATES[component],
        "suggested_llm_prompt": suggested_prompt,
        "layout_geometry": {
            "aspect_ratio": round(geom["aspect_ratio"], 3),
            "normalized_area": round(geom["normalized_area"], 3),
            "horizontal_bounds": geom["horizontal_bounds"],
            "vertical_bounds": geom["vertical_bounds"],
            "deskew_angle": geom["deskew_angle"],
            "stroke_density": round(geom["ink_density"], 4),
            "maximal_histogram_rectangle": {"max_area": max_area, "col_range": [left_col, right_col], "height": max_h},
            "connected_stroke_islands": len(stroke_islands),
            "nms_merged_intervals": merged_intervals,
        },
        "jsx_hierarchy": jsx_hierarchy,
        "template_edit_distance": template_edit_dist,
        "all_algorithms_executed": {
            "lab0_linear_algebra": f"Deskewed by {geom['deskew_angle']:.1f} deg with orthogonal R (det={geom['rotation_det']:.2f}, trace={geom['rotation_trace']:.2f})",
            "lab1_tf_graph": f"Tensor arithmetic (Add={tensor_add:.2f}, Sub={tensor_sub:.2f})",
            "lab2_scratch_backprop": f"Scratch MLP forward pass: {[round(p, 3) for p in calib_out]}",
            "lab3_batch_norm": "Pixel-batch normalization (zero mean, unit variance) before AlexNet",
            "lab4_perceptron_boundary": f"Hyperplane gate: '{layout_gate_decision}' (AND gate -w0={minus_w0_val})",
            "lab5_style_ann": f"Dropout ANN raw output '{style_ann_out['predicted_variant']}' (untrained; variant chosen by geometry rules)",
            "lab7_alexnet_lrn": f"{classifier_desc}: '{component}' ({confidence:.1%})",
            "fraz_2d_prefix_sum": f"O(1) ink density query = {geom['ink_density']:.4f}",
            "fraz_two_pointers": f"Tight bounds x={geom['horizontal_bounds']} y={geom['vertical_bounds']}",
            "fraz_monotonic_stack": f"Largest Rectangle in Histogram (LC 84): Area={max_area}",
            "fraz_trapping_rain_water": f"Trapped negative space (LC 42) = {trapped_water}",
            "fraz_container_water": f"Container With Most Water (LC 11) = {max_water}",
            "fraz_sliding_window_max": f"Sliding window max peak (LC 239) = {max(smoothed_peaks, default=0)}",
            "fraz_subarray_sum_k": f"Column runs summing to 5 (LC 560) = {divider_count}",
            "fraz_merge_intervals": f"NMS merged island intervals (LC 56): {merged_intervals}",
            "fraz_graph_bfs_islands": f"Connected stroke components (LC 200): {len(stroke_islands)} islands",
            "fraz_kahn_topological_sort": f"JSX order (LC 210): {' -> '.join(jsx_hierarchy)} (tags valid: {is_jsx_valid})",
            "fraz_dijkstra": f"Routing distance PageWrapper -> {component} (LC 743) = {route_dist[component]}",
            "fraz_tree_lca": f"LCA(Container, {component}) (LC 236) = {lca.val if lca else 'None'}",
            "fraz_binary_search": f"Confidence cutoff index = {cutoff_rank}, viewport capacity (LC 1011) = {viewport_cap}",
            "fraz_knapsack": f"Viewport knapsack priority = {packed_priority}",
            "fraz_trie_autocomplete": f"Trie suggestions: {tailwind_classes[:3]}",
            "fraz_top_k_heaps": f"Min-heap Top-3 classes: {[c for c, _ in top_candidates]}",
            "fraz_dp_edit_distance": f"Levenshtein distance to design template (LC 72): {template_edit_dist}",
        },
        "processing_time_ms": f"{elapsed_ms:.1f}ms",
    }

  @staticmethod
  def _empty_result(start_time: float) -> dict:
    return {
        "status": "empty",
        "detected_component": "Empty Canvas",
        "confidence": "0.00%",
        "classifier": "none",
        "style_variant": "None",
        "top_3_candidates": [],
        "generated_react_code": (
            "// Canvas is empty.\n"
            "// Please draw a UI sketch (Button, Card, InputField, Navbar, Modal)\n"
            "// or click one of the pre-made sample buttons to generate React code."
        ),
        "suggested_llm_prompt": "Canvas is empty. Draw a UI wireframe or click a sample to generate an enriched prompt.",
        "layout_geometry": {},
        "jsx_hierarchy": [],
        "template_edit_distance": 0,
        "all_algorithms_executed": {},
        "processing_time_ms": f"{(time.time() - start_time) * 1000.0:.1f}ms",
    }


_pipeline_instance = None


def get_pipeline() -> Sketch2ReactPipeline:
  global _pipeline_instance
  if _pipeline_instance is None:
    _pipeline_instance = Sketch2ReactPipeline()
  return _pipeline_instance
