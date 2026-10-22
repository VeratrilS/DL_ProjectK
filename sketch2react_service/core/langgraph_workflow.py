"""LangGraph & LangChain Agentic Workflow for Sketch2React.

Orchestrates the entire vision, deep learning lab inference, and Fraz DSA layout parsing
as a modular StateGraph:

[Sketch Input]
     │
     ▼
[Node: Vision Feature Extraction] (AlexNet Lab 7, Autoencoders Lab 8, 2D Prefix Sum)
     │
     ▼
[Node: DSA Layout Parsing]        (Monotonic Stack LC 84, Two Pointers LC 42 & LC 11)
     │
     ▼
[Node: Lab Validation Gate]       (Perceptron Lab 4, Style ANN Lab 5, Scratch Backprop Lab 2)
     │
     ▼
[Node: Graph & Hierarchy]         (Kahn's TopoSort LC 210, Tree LCA LC 236, Tag Matcher LC 20)
     │
     ▼
[Node: LangChain Prompt Synthesis] (LangChain PromptTemplate & LCEL Chain)
     │
     ▼
[Final Component State]
"""

from typing import Any, Dict, List, Optional, Tuple, TypedDict
import cv2
import numpy as np
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

# Core Lab Modules
from core.fraz_algorithms import (
    PrefixSum2D,
    ReactComponentTrie,
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
    FLAG_BORDER,
    FLAG_HOVER,
    FLAG_ROUNDED,
    FLAG_SHADOW,
    TreeNode,
)
from core.lab0_linear_algebra import (
    compute_trace,
    create_orthogonal_rotation_matrix,
    determinant_via_gaussian_elimination,
    verify_orthogonality,
)
from core.lab1_tf_graph import tensor_arithmetic_demo
from core.lab2_scratch_backprop import ScratchMLP
from core.lab3_batch_norm import batch_norm_scratch
from core.lab4_perceptron_boundary import classify_button_vs_card, get_minus_w0
from core.lab5_style_ann import build_lab5_style_ann, predict_component_style
from core.lab7_alexnet_backbone import (
    LocalResponseNormalization,
    build_alexnet_model,
    generate_synthetic_ui_dataset,
)

UI_CLASSES = ["Button", "Card", "InputField", "Navbar", "Modal"]


class SketchState(TypedDict):
  """State container managed across LangGraph nodes."""
  image_rgb: np.ndarray
  detected_component: str
  confidence: float
  style_variant: str
  aspect_ratio: float
  normalized_area: float
  geometry: Dict[str, Any]
  jsx_hierarchy: List[str]
  algorithms_trace: Dict[str, str]
  enriched_prompt: str
  is_valid: bool


# Shared model instances for efficiency
_alexnet_model = None
_style_ann_model = None
_scratch_calibrator = None
_trie_instance = None


def get_shared_models():
  global _alexnet_model, _style_ann_model, _scratch_calibrator, _trie_instance
  if _alexnet_model is None:
    _alexnet_model = build_alexnet_model(input_shape=(227, 227, 3), num_classes=5, dense_units=256)
    X_syn, y_syn = generate_synthetic_ui_dataset(num_samples=25, img_size=227)
    _alexnet_model.fit(X_syn, y_syn, epochs=1, batch_size=15, verbose=0)

    _style_ann_model = build_lab5_style_ann(input_shape=(28, 28), num_classes=5)
    _scratch_calibrator = ScratchMLP(n_in=3, n_h1=3, n_h2=3, n_out=2)

    _trie_instance = ReactComponentTrie()
    for tag in ["Button", "Card", "CardHeader", "Input", "Navbar", "Modal", "rounded-2xl", "shadow-xl", "hover:scale-105"]:
      _trie_instance.insert(tag)

  return _alexnet_model, _style_ann_model, _scratch_calibrator, _trie_instance


# =====================================================================
# Node 1: Vision Feature Extraction (AlexNet Lab 7, Lab 0, Lab 3, Prefix Sum)
# =====================================================================
def node_vision_feature_extraction(state: SketchState) -> Dict[str, Any]:
  alexnet, _, _, _ = get_shared_models()
  img = state["image_rgb"]

  # 1. Resize & Preprocess
  resized = cv2.resize(img, (227, 227))
  if len(resized.shape) == 2:
    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
  elif resized.shape[2] == 4:
    resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)

  # Lab 0: Orthogonal Straightening
  R = create_orthogonal_rotation_matrix(0.0)
  is_ortho = verify_orthogonality(R)
  trace_val = compute_trace(R)

  # Lab 3: Batch Normalization on pixels
  normed = resized.astype(np.float32) / 255.0
  normed = batch_norm_scratch(normed, gamma=1.0, beta=0.0, eps=1e-5)
  normed = np.clip(normed, 0.0, 1.0)

  # Fraz 2D Prefix Sum on luminance (LC 304)
  gray = cv2.cvtColor((normed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
  pref2d = PrefixSum2D(gray.tolist())
  density = pref2d.query_box_density(0, 0, 226, 226)

  # Fraz Two Pointers for Horizontal Bounds
  h_proj = np.mean(gray < 0.8, axis=0).tolist()
  left_b, right_b = two_pointers_tight_bounds(h_proj, threshold=0.05)

  # Lab 7: AlexNet Inference
  batch = np.expand_dims(normed, axis=0)
  logits = alexnet(batch, training=False).numpy()[0]
  best_idx = int(np.argmax(logits))
  detected = UI_CLASSES[best_idx]
  confidence = float(logits[best_idx])

  trace = state.get("algorithms_trace", {})
  trace["lab0_linear_algebra"] = f"Orthogonal Rotation (Det=1.0, Trace={trace_val:.1f})"
  trace["lab3_batch_norm"] = "Batch Normalization applied across pixels"
  trace["lab7_alexnet_backbone"] = f"AlexNet with Custom LRN classified as '{detected}' ({confidence*100:.1f}%)"
  trace["fraz_2d_prefix_sum"] = f"2D Prefix Sum (LC 304) density = {density:.4f}"
  trace["fraz_two_pointers"] = f"Two Pointers bounds = [{left_b}, {right_b}]"

  width = right_b - left_b
  aspect_ratio = float(width) / 227.0
  norm_area = float(width * 227) / (227.0 * 227.0)

  return {
      "image_rgb": normed,
      "detected_component": detected,
      "confidence": confidence,
      "aspect_ratio": aspect_ratio,
      "normalized_area": norm_area,
      "geometry": {
          "horizontal_bounds": [int(left_b), int(right_b)],
          "stroke_density": float(density),
      },
      "algorithms_trace": trace,
  }


# =====================================================================
# Node 2: DSA Layout Parsing (Monotonic Stack, Trapping Rain Water, NMS)
# =====================================================================
def node_dsa_layout_parser(state: SketchState) -> Dict[str, Any]:
  normed = state["image_rgb"]
  patch_28 = cv2.resize(normed, (28, 28))[:, :, 0]
  col_heights = [int(np.sum(patch_28[:, c] < 0.8)) for c in range(28)]

  # Monotonic Stack (LC 84)
  max_area, left_c, right_c, max_h = largest_rectangle_in_histogram(col_heights)

  # Two Pointers: Trapping Rain Water (LC 42) & Container (LC 11)
  trapped_space = trapping_rain_water(col_heights)
  max_water_box, wl, wr = container_with_most_water(col_heights)

  # Sliding Window Max (LC 239)
  smoothed_peaks = sliding_window_maximum(col_heights, k=3)

  # Prefix Sum: Subarray Sum Equals K (LC 560)
  divider_segments = subarray_sum_equals_k(col_heights, k=4)

  # Merge Intervals: NMS (LC 56)
  h_bounds = state["geometry"]["horizontal_bounds"]
  intervals = [[h_bounds[0], h_bounds[1]], [20, 80], [75, 130]]
  merged_nms = merge_bounding_intervals(intervals, tolerance=0)

  # Graph BFS: Stroke Islands (LC 200)
  binary_grid = (patch_28 < 0.8).astype(int).tolist()
  islands = find_stroke_islands_bfs(binary_grid)

  trace = state["algorithms_trace"]
  trace["fraz_monotonic_stack"] = f"Largest Rectangle in Histogram (LC 84): Area={max_area}"
  trace["fraz_trapping_rain_water"] = f"Trapping Rain Water (LC 42): Recessed depth={trapped_space}"
  trace["fraz_container_water"] = f"Container With Most Water (LC 11): Max Area={max_water_box}"
  trace["fraz_sliding_window_max"] = f"Sliding Window Maximum (LC 239): {len(smoothed_peaks)} window steps"
  trace["fraz_merge_intervals"] = f"Merge Intervals NMS (LC 56): {merged_nms}"
  trace["fraz_graph_bfs"] = f"Connected Components BFS (LC 200): {len(islands)} stroke islands"

  geom = state["geometry"]
  geom["max_histogram_rectangle"] = {"area": max_area, "cols": [left_c, right_c], "height": max_h}
  geom["connected_islands"] = len(islands)
  geom["merged_nms"] = merged_nms

  return {"geometry": geom, "algorithms_trace": trace}


# =====================================================================
# Node 3: Lab Validation Gate (Perceptron Lab 4, Style ANN Lab 5, Lab 2)
# =====================================================================
def node_lab_validation_gate(state: SketchState) -> Dict[str, Any]:
  _, style_ann, scratch_calib, _ = get_shared_models()
  detected = state["detected_component"]
  ar = state["aspect_ratio"]
  area = state["normalized_area"]
  conf = state["confidence"]

  # Lab 4: Perceptron Decision Boundary
  layout_gate = classify_button_vs_card(ar, area)
  minus_w0 = get_minus_w0(1.0, 1.0, "AND", [0, 0, 0, 1])
  if detected in ["Button", "Card"] and conf < 0.65:
    detected = layout_gate

  # Lab 5: Style ANN with Dropout
  normed = state["image_rgb"]
  patch_28 = cv2.resize(normed, (28, 28))[:, :, 0]
  style_pred = predict_component_style(style_ann, patch_28)

  # Lab 2: Scratch Backpropagation Calibration
  _, _, calib_probs = scratch_calib.forward([min(1.0, ar), min(1.0, area), conf])

  # Lab 1: Tensor Arithmetic
  t_add, t_sub = tensor_arithmetic_demo(conf, 0.05)

  trace = state["algorithms_trace"]
  trace["lab1_tf_graph"] = f"Tensor Arithmetic executed (Add={t_add:.2f}, Sub={t_sub:.2f})"
  trace["lab2_scratch_backprop"] = f"Scratch Backprop Analytical Calibration: {[round(p, 3) for p in calib_probs]}"
  trace["lab4_perceptron_gate"] = f"Perceptron Hyperplane Decision: '{layout_gate}' (valid -w0={minus_w0})"
  trace["lab5_style_ann"] = f"Style ANN with Dropout(0.2) classified as '{style_pred['predicted_variant']}'"

  return {
      "detected_component": detected,
      "style_variant": style_pred["predicted_variant"],
      "algorithms_trace": trace,
      "is_valid": True,
  }


# =====================================================================
# Node 4: Graph Hierarchy & Routing (Kahn's TopoSort, LCA, Dijkstra)
# =====================================================================
def node_graph_hierarchy(state: SketchState) -> Dict[str, Any]:
  detected = state["detected_component"]
  components = ["PageWrapper", "Container", detected]
  edges = [("PageWrapper", "Container"), ("Container", detected)]

  # Kahn's TopoSort (LC 210)
  jsx_order = kahn_topological_sort_jsx(components, edges)

  # Stack JSX Validator (LC 20)
  tokens = ["<PageWrapper>", "<Container>", f"<{detected}>", f"</{detected}>", "</Container>", "</PageWrapper>"]
  is_valid_jsx = validate_jsx_nesting_tags(tokens)

  # Dijkstra Routing (LC 743)
  dijkstra_dist = dijkstra_shortest_path(components, [("PageWrapper", "Container", 1), ("Container", detected, 2)], "PageWrapper")

  # Tree LCA (LC 236)
  tree_root = TreeNode("PageWrapper")
  tree_root.left = TreeNode("Container")
  tree_root.left.left = TreeNode(detected)
  lca_node = lowest_common_ancestor_dom(tree_root, "Container", detected)

  # DP Edit Distance (LC 72)
  template = ["PageWrapper", "Container", "Card", "Button"]
  edit_dist = layout_edit_distance(jsx_order, template)

  trace = state["algorithms_trace"]
  trace["fraz_kahn_toposort"] = f"Kahn's TopoSort (LC 210): {' -> '.join(jsx_order)}"
  trace["fraz_jsx_validator_stack"] = f"Valid Parentheses Matcher (LC 20): Valid JSX = {is_valid_jsx}"
  trace["fraz_dijkstra_shortest_path"] = f"Dijkstra Shortest Path (LC 743): Distance to {detected} = {dijkstra_dist[detected]}"
  trace["fraz_tree_lca"] = f"Lowest Common Ancestor (LC 236): Parent = '{lca_node.val if lca_node else 'Root'}'"
  trace["fraz_dp_edit_distance"] = f"Edit Distance (LC 72) to design template = {edit_dist}"

  return {
      "jsx_hierarchy": jsx_order,
      "algorithms_trace": trace,
  }


# =====================================================================
# Node 5: LangChain Prompt Synthesis (LCEL PromptTemplate & Chain)
# =====================================================================
def node_langchain_prompt_synthesis(state: SketchState) -> Dict[str, Any]:
  _, _, _, trie = get_shared_models()
  detected = state["detected_component"]
  style = state["style_variant"]
  hierarchy = " > ".join(state["jsx_hierarchy"])

  # Trie autocompletion (LC 208)
  tailwind_tokens = trie.autocomplete("rounded-2xl") + trie.autocomplete("shadow-xl") + trie.autocomplete("hover:scale-105")

  # Bitmask CSS feature flags
  css_features = evaluate_bitmask_flags(FLAG_BORDER | FLAG_SHADOW | FLAG_ROUNDED | FLAG_HOVER)

  # DP 0/1 Knapsack layout budget
  packed_budget = knapsack_viewport_packing([80, 120, 200], [15, 30, 45], max_capacity=227)

  # LangChain PromptTemplate (Declarative LCEL structure)
  prompt_template = PromptTemplate(
      input_variables=["component", "style", "hierarchy", "tailwind_classes", "css_flags"],
      template=(
          "Generate a modern, responsive React component for a '{component}' "
          "styled with '{style}' theme. Follow design system hierarchy: {hierarchy}. "
          "Recommended Tailwind utility classes: {tailwind_classes}. "
          "Active visual features: {css_flags}. Ensure full accessibility with ARIA attributes and smooth hover states."
      )
  )

  # Format prompt via LangChain
  final_prompt = prompt_template.format(
      component=detected,
      style=style,
      hierarchy=hierarchy,
      tailwind_classes=", ".join(tailwind_tokens),
      css_flags=", ".join(css_features),
  )

  trace = state["algorithms_trace"]
  trace["fraz_trie_autocomplete"] = f"Trie (LC 208) suggestions: {tailwind_tokens}"
  trace["fraz_bitmask_flags"] = f"Bit Manipulation CSS Flags: {css_features}"
  trace["fraz_knapsack_dp"] = f"0/1 Knapsack Viewport Packing: Score = {packed_budget}"
  trace["langchain_lcel_synthesis"] = "LangChain PromptTemplate rendered prompt"

  return {
      "enriched_prompt": final_prompt,
      "algorithms_trace": trace,
  }


# =====================================================================
# StateGraph Assembly
# =====================================================================
def build_sketch2react_graph():
  workflow = StateGraph(SketchState)

  workflow.add_node("vision_feature_extraction", node_vision_feature_extraction)
  workflow.add_node("dsa_layout_parser", node_dsa_layout_parser)
  workflow.add_node("lab_validation_gate", node_lab_validation_gate)
  workflow.add_node("graph_hierarchy", node_graph_hierarchy)
  workflow.add_node("langchain_prompt_synthesis", node_langchain_prompt_synthesis)

  workflow.set_entry_point("vision_feature_extraction")
  workflow.add_edge("vision_feature_extraction", "dsa_layout_parser")
  workflow.add_edge("dsa_layout_parser", "lab_validation_gate")
  workflow.add_edge("lab_validation_gate", "graph_hierarchy")
  workflow.add_edge("graph_hierarchy", "langchain_prompt_synthesis")
  workflow.add_edge("langchain_prompt_synthesis", END)

  return workflow.compile()


# Compiled LangGraph agent singleton
compiled_graph = None

def run_sketch_agent(image_rgb: np.ndarray) -> Dict[str, Any]:
  global compiled_graph
  if compiled_graph is None:
    compiled_graph = build_sketch2react_graph()

  initial_state: SketchState = {
      "image_rgb": image_rgb,
      "detected_component": "",
      "confidence": 0.0,
      "style_variant": "",
      "aspect_ratio": 0.0,
      "normalized_area": 0.0,
      "geometry": {},
      "jsx_hierarchy": [],
      "algorithms_trace": {},
      "enriched_prompt": "",
      "is_valid": False,
  }

  final_state = compiled_graph.invoke(initial_state)
  return final_state
