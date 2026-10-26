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

from typing import Any, Dict, List, TypedDict

import cv2
import numpy as np
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

from core.fraz_algorithms import (
    FLAG_BORDER,
    FLAG_HOVER,
    FLAG_ROUNDED,
    FLAG_SHADOW,
    TreeNode,
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
    trapping_rain_water,
    validate_jsx_nesting_tags,
)
from core.lab1_tf_graph import tensor_arithmetic_demo
from core.lab4_perceptron_boundary import classify_button_vs_card, get_minus_w0
from core.lab5_style_ann import predict_component_style
from pipeline import MIN_INK_PIXELS, get_pipeline, style_variant_for


class SketchState(TypedDict):
  """State container managed across LangGraph nodes."""
  image_rgb: np.ndarray  # input sketch; replaced by the 227x227 ink map after node 1
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


# =====================================================================
# Node 1: Vision Feature Extraction (Lab 0 deskew, Lab 3 BN, Lab 7 AlexNet, Prefix Sum, Two Pointers)
# =====================================================================
def node_vision_feature_extraction(state: SketchState) -> Dict[str, Any]:
  pipe = get_pipeline()
  gray, ink, geom = pipe.preprocess_sketch_image(state["image_rgb"])
  if geom["total_ink_pixels"] < MIN_INK_PIXELS:
    raise ValueError("Sketch is empty: draw a UI component first.")

  layout_gate = classify_button_vs_card(geom["aspect_ratio"], geom["normalized_area"])
  detected, probs, source = pipe.classify(gray, geom, layout_gate)

  trace = state.get("algorithms_trace", {})
  trace["lab0_linear_algebra"] = f"Deskewed by {geom['deskew_angle']:.1f} deg (det={geom['rotation_det']:.2f}, trace={geom['rotation_trace']:.2f})"
  trace["lab3_batch_norm"] = "Pixel-batch normalization applied before AlexNet"
  trace["lab7_alexnet_backbone"] = f"{source} classified as '{detected}' ({probs[detected]:.1%})"
  trace["fraz_2d_prefix_sum"] = f"2D Prefix Sum (LC 304) density = {geom['ink_density']:.4f}"
  trace["fraz_two_pointers"] = f"Two Pointers bounds x={geom['horizontal_bounds']} y={geom['vertical_bounds']}"

  return {
      "image_rgb": ink,
      "detected_component": detected,
      "confidence": probs[detected],
      "aspect_ratio": geom["aspect_ratio"],
      "normalized_area": geom["normalized_area"],
      "geometry": geom,
      "algorithms_trace": trace,
  }


# =====================================================================
# Node 2: DSA Layout Parsing (Monotonic Stack, Trapping Rain Water, NMS)
# =====================================================================
def node_dsa_layout_parser(state: SketchState) -> Dict[str, Any]:
  patch_28 = cv2.resize(state["image_rgb"], (28, 28), interpolation=cv2.INTER_AREA)
  binary_grid = (patch_28 > 0.1).astype(int).tolist()
  col_heights = [int(v) for v in np.sum(np.array(binary_grid), axis=0)]

  # Monotonic Stack (LC 84)
  max_area, left_c, right_c, max_h = largest_rectangle_in_histogram(col_heights)

  # Two Pointers: Trapping Rain Water (LC 42) & Container (LC 11)
  trapped_space = trapping_rain_water(col_heights)
  max_water_box, wl, wr = container_with_most_water(col_heights)

  # Sliding Window Max (LC 239)
  smoothed_peaks = sliding_window_maximum(col_heights, k=3)

  # Prefix Sum: Subarray Sum Equals K (LC 560)
  divider_segments = subarray_sum_equals_k(col_heights, k=4)

  # Graph BFS: Stroke Islands (LC 200)
  islands = find_stroke_islands_bfs(binary_grid)

  # Merge Intervals: NMS over the islands' horizontal extents (LC 56)
  scale = 227 / 28.0
  intervals = [[int(min(c for _, c in isl) * scale), int((max(c for _, c in isl) + 1) * scale)] for isl in islands]
  merged_nms = merge_bounding_intervals(intervals, tolerance=int(scale))

  trace = state["algorithms_trace"]
  trace["fraz_monotonic_stack"] = f"Largest Rectangle in Histogram (LC 84): Area={max_area}"
  trace["fraz_trapping_rain_water"] = f"Trapping Rain Water (LC 42): Recessed depth={trapped_space}"
  trace["fraz_container_water"] = f"Container With Most Water (LC 11): Max Area={max_water_box}"
  trace["fraz_sliding_window_max"] = f"Sliding Window Maximum (LC 239): {len(smoothed_peaks)} window steps"
  trace["fraz_subarray_sum_k"] = f"Subarray Sum Equals K (LC 560): {divider_segments} segments"
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
  pipe = get_pipeline()
  detected = state["detected_component"]
  ar = state["aspect_ratio"]
  area = state["normalized_area"]
  conf = state["confidence"]

  # Lab 4: Perceptron Decision Boundary
  layout_gate = classify_button_vs_card(ar, area)
  minus_w0 = get_minus_w0(1.0, 1.0, "AND", [0, 0, 0, 1])

  # Lab 5: Style ANN (untrained) + geometric style rules, same as the REST pipeline
  patch_28 = cv2.resize(state["image_rgb"], (28, 28), interpolation=cv2.INTER_AREA)
  style_pred = predict_component_style(pipe.style_ann, patch_28)
  style_variant = style_variant_for(detected, state["geometry"]["ink_density"], ar)

  # Lab 2: Scratch MLP forward pass
  _, _, calib_probs = pipe.scratch_calibrator.forward([min(1.0, ar / 4.0), min(1.0, area), conf])

  # Lab 1: Tensor Arithmetic
  t_add, t_sub = tensor_arithmetic_demo(conf, 0.05)

  trace = state["algorithms_trace"]
  trace["lab1_tf_graph"] = f"Tensor Arithmetic executed (Add={t_add:.2f}, Sub={t_sub:.2f})"
  trace["lab2_scratch_backprop"] = f"Scratch MLP forward pass: {[round(p, 3) for p in calib_probs]}"
  trace["lab4_perceptron_gate"] = f"Perceptron Hyperplane Decision: '{layout_gate}' (valid -w0={minus_w0})"
  trace["lab5_style_ann"] = f"Dropout ANN raw output '{style_pred['predicted_variant']}' (untrained; variant '{style_variant}' from geometry rules)"

  return {
      "style_variant": style_variant,
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
  trie = get_pipeline().trie
  detected = state["detected_component"]
  style = state["style_variant"]
  hierarchy = " > ".join(state["jsx_hierarchy"])

  # Trie autocompletion (LC 208)
  tailwind_tokens = trie.autocomplete("rounded") + trie.autocomplete("shadow") + trie.autocomplete("hover:")

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
