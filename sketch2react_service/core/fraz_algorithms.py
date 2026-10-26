"""Fraz's DSA Sheet Master Algorithmic Suite for Sketch2React.

Contains 18 Classic Algorithmic Patterns from Fraz's Sheet used in UI Layout Parsing:
1.  Prefix Sum 2D (LC 304) -> O(1) Bounding Box Stroke Density
2.  Two Pointers (LC 11 Container With Most Water) -> Canvas Border Boundaries
3.  Two Pointers (LC 42 Trapping Rain Water) -> UI Recessed Well / Input Field Depth
4.  Sliding Window Maximum (LC 239 Monotonic Deque) -> Stroke Signal Peak Smoothing
5.  Prefix Sum Subarray Sum Equals K (LC 560) -> Detecting Continuous Divider Lines
6.  Stack: Largest Rectangle in Histogram (LC 84) -> Maximal Container Detection
7.  Stack: Valid Parentheses / JSX Tag Matcher (LC 20) -> Validating React DOM Trees
8.  Greedy Intervals: Merge Intervals (LC 56) -> Non-Maximum Suppression (NMS)
9.  Graph BFS: Number of Islands (LC 200) -> Connected Component Stroke Islands
10. Graph DSU: Disjoint Set Union (LC 684) -> Clustering Proximity Stroke Fragments
11. Graph TopoSort: Kahn's Algorithm (LC 210) -> JSX Parent-to-Child Rendering Order
12. Graph Dijkstra: Shortest Path (LC 743) -> Routing UI Connector Arrows/Lines
13. Tree LCA: Lowest Common Ancestor (LC 236) -> Finding Shared Parent Container in DOM
14. Binary Search on Answer: Capacity Allocation (LC 1011) -> Dynamic Viewport Grid Fitting
15. Trie: Prefix Tree (LC 208) -> React Tag & Tailwind CSS Class Autocompletion
16. Heap / Priority Queue: Top-K Frequent (LC 347) -> Top-K Confident UI Predictions
17. Dynamic Programming: 0/1 Knapsack Layout Packing -> Viewport Budget Component Fitting
18. Dynamic Programming: Edit Distance (LC 72) -> Layout Tree Design Alignment
19. Bit Manipulation: Bitmask Flags -> Component Style Attribute Toggles
"""

import collections
import heapq
from typing import Dict, List, Optional, Set, Tuple


# =====================================================================
# 1. 2D PREFIX SUM: Range Sum Query 2D (LeetCode 304)
# =====================================================================
class PrefixSum2D:
  """2D Prefix Sum for O(1) query of stroke pixel density in any bounding box [r1..r2, c1..c2].

  Time: O(R*C) Precomputation, O(1) per Query
  """

  def __init__(self, matrix: List[List[float]]):
    R, C = len(matrix), len(matrix[0])
    self.pref = [[0.0] * (C + 1) for _ in range(R + 1)]
    for r in range(R):
      for c in range(C):
        self.pref[r + 1][c + 1] = (
            matrix[r][c]
            + self.pref[r][c + 1]
            + self.pref[r + 1][c]
            - self.pref[r][c]
        )

  def query_box_density(self, r1: int, c1: int, r2: int, c2: int) -> float:
    total = (
        self.pref[r2 + 1][c2 + 1]
        - self.pref[r1][c2 + 1]
        - self.pref[r2 + 1][c1]
        + self.pref[r1][c1]
    )
    area = (r2 - r1 + 1) * (c2 - c1 + 1)
    return total / area if area > 0 else 0.0


# =====================================================================
# 2. TWO POINTERS: Container With Most Water (LeetCode 11)
# =====================================================================
def container_with_most_water(heights: List[int]) -> Tuple[int, int, int]:
  """Finds two vertical stroke boundaries that enclose the maximum canvas area.

  Fraz Module: Two Pointers (LC 11)
  Time: O(N), Space: O(1)
  """
  left, right = 0, len(heights) - 1
  max_water = 0
  best_left, best_right = 0, len(heights) - 1

  while left < right:
    h = min(heights[left], heights[right])
    width = right - left
    current_area = h * width
    if current_area > max_water:
      max_water = current_area
      best_left, best_right = left, right

    if heights[left] < heights[right]:
      left += 1
    else:
      right -= 1

  return max_water, best_left, best_right


# =====================================================================
# 3. TWO POINTERS: Trapping Rain Water (LeetCode 42)
# =====================================================================
def trapping_rain_water(profile: List[int]) -> int:
  """Calculates trapped visual negative space (detecting recessed input fields & card wells).

  Fraz Module: Two Pointers (LC 42)
  Time: O(N), Space: O(1)
  """
  if not profile:
    return 0

  left, right = 0, len(profile) - 1
  left_max, right_max = profile[left], profile[right]
  trapped_water = 0

  while left < right:
    if profile[left] < profile[right]:
      if profile[left] >= left_max:
        left_max = profile[left]
      else:
        trapped_water += left_max - profile[left]
      left += 1
    else:
      if profile[right] >= right_max:
        right_max = profile[right]
      else:
        trapped_water += right_max - profile[right]
      right -= 1

  return trapped_water


# =====================================================================
# 4. TWO POINTERS: 1D Projection Tight Bounds
# =====================================================================
def two_pointers_tight_bounds(profile: List[float], threshold: float = 0.05) -> Tuple[int, int]:
  """Uses Two Pointers converging from left and right to find tight component bounds.

  Time: O(N), Space: O(1)
  """
  left, right = 0, len(profile) - 1
  while left < len(profile) and profile[left] <= threshold:
    left += 1
  while right >= 0 and profile[right] <= threshold:
    right -= 1
  return (left, right) if left <= right else (0, len(profile) - 1)


# =====================================================================
# 5. SLIDING WINDOW: Sliding Window Maximum (LeetCode 239 Monotonic Deque)
# =====================================================================
def sliding_window_maximum(nums: List[int], k: int) -> List[int]:
  """Computes running maximum stroke elevation using a Monotonic Decreasing Deque.

  Fraz Module: Sliding Window / Deque (LC 239)
  Time: O(N), Space: O(K)
  """
  if not nums or k <= 0:
    return []

  q = collections.deque()  # Stores indices
  result = []

  for i, n in enumerate(nums):
    # Remove indices out of current window
    while q and q[0] < i - k + 1:
      q.popleft()

    # Maintain monotonic decreasing order
    while q and nums[q[-1]] < n:
      q.pop()

    q.append(i)

    if i >= k - 1:
      result.append(nums[q[0]])

  return result


# =====================================================================
# 6. PREFIX SUM: Subarray Sum Equals K (LeetCode 560)
# =====================================================================
def subarray_sum_equals_k(nums: List[int], k: int) -> int:
  """Counts continuous pixel segments matching exact stroke width K using Hash Map + Prefix Sum.

  Fraz Module: Prefix Sum (LC 560)
  Time: O(N), Space: O(N)
  """
  prefix_counts = collections.defaultdict(int)
  prefix_counts[0] = 1
  curr_sum = 0
  total_matches = 0

  for num in nums:
    curr_sum += num
    if (curr_sum - k) in prefix_counts:
      total_matches += prefix_counts[curr_sum - k]
    prefix_counts[curr_sum] += 1

  return total_matches


# =====================================================================
# 7. MONOTONIC STACK: Largest Rectangle in Histogram (LeetCode 84)
# =====================================================================
def largest_rectangle_in_histogram(heights: List[int]) -> Tuple[int, int, int, int]:
  """Finds maximal bounding box rectangle in histogram using Monotonic Stack.

  Fraz Module: Stack (LC 84)
  Time: O(N), Space: O(N)
  """
  stack = []
  max_area = 0
  best_box = (0, 0, 0, 0)
  extended = heights + [0]

  for i, h in enumerate(extended):
    start = i
    while stack and stack[-1][1] > h:
      idx, prev_h = stack.pop()
      width = i - idx
      area = prev_h * width
      if area > max_area:
        max_area = area
        best_box = (max_area, idx, i - 1, prev_h)
      start = idx
    stack.append((start, h))

  return best_box


# =====================================================================
# 8. STACK: Valid Parentheses / JSX Tag Matcher (LeetCode 20)
# =====================================================================
def validate_jsx_nesting_tags(tokens: List[str]) -> bool:
  """Validates if generated React JSX opening and closing component tags match correctly.

  Fraz Module: Stack (LC 20)
  Time: O(N), Space: O(N)
  """
  stack = []
  for token in tokens:
    if token.endswith("/>"):
      continue  # self-closing tag like <Button /> opens and closes itself
    if token.startswith("<") and not token.startswith("</"):
      tag_name = token.strip("<>")
      stack.append(tag_name)
    elif token.startswith("</"):
      tag_name = token.strip("</>")
      if not stack or stack[-1] != tag_name:
        return False
      stack.pop()

  return len(stack) == 0


# =====================================================================
# 9. GREEDY / INTERVALS: Merge Intervals (LeetCode 56)
# =====================================================================
def merge_bounding_intervals(intervals: List[List[int]], tolerance: int = 0) -> List[List[int]]:
  """Non-Maximum Suppression (NMS) merging overlapping bounding box intervals.

  Fraz Module: Intervals & Greedy (LC 56)
  Time: O(N log N), Space: O(N)
  """
  if not intervals:
    return []

  sorted_iv = sorted([list(iv) for iv in intervals], key=lambda x: x[0])
  merged = [sorted_iv[0]]

  for current in sorted_iv[1:]:
    last = merged[-1]
    if current[0] <= last[1] + tolerance:
      last[1] = max(last[1], current[1])
    else:
      merged.append(current)

  return merged


# =====================================================================
# 10. GRAPH BFS: Connected Components / Number of Islands (LeetCode 200)
# =====================================================================
def find_stroke_islands_bfs(grid: List[List[int]]) -> List[List[Tuple[int, int]]]:
  """Groups connected drawn sketch pixels into isolated UI component stroke islands.

  Fraz Module: Graphs BFS (LC 200)
  Time: O(R * C), Space: O(R * C)
  """
  R, C = len(grid), len(grid[0])
  visited = set()
  islands = []

  for r in range(R):
    for c in range(C):
      if grid[r][c] == 1 and (r, c) not in visited:
        queue = collections.deque([(r, c)])
        visited.add((r, c))
        current_island = []

        while queue:
          curr_r, curr_c = queue.popleft()
          current_island.append((curr_r, curr_c))

          for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 1 and (nr, nc) not in visited:
              visited.add((nr, nc))
              queue.append((nr, nc))

        islands.append(current_island)

  return islands


# =====================================================================
# 11. GRAPH DSU: Disjoint Set Union with Path Compression & Rank (LeetCode 684)
# =====================================================================
class DisjointSetUnion:
  """Union-Find / Disjoint Set Union (DSU) for clustering nearby stroke bounding boxes.

  Fraz Module: Graphs / DSU (LC 684)
  Time: O(alpha(N)) nearly O(1) per operation
  """

  def __init__(self, n: int):
    self.parent = list(range(n))
    self.rank = [0] * n

  def find(self, i: int) -> int:
    if self.parent[i] != i:
      self.parent[i] = self.find(self.parent[i])  # Path compression
    return self.parent[i]

  def union(self, i: int, j: int) -> bool:
    root_i = self.find(i)
    root_j = self.find(j)
    if root_i == root_j:
      return False  # Already connected

    # Union by rank
    if self.rank[root_i] < self.rank[root_j]:
      self.parent[root_i] = root_j
    elif self.rank[root_i] > self.rank[root_j]:
      self.parent[root_j] = root_i
    else:
      self.parent[root_j] = root_i
      self.rank[root_i] += 1
    return True


# =====================================================================
# 12. GRAPH TOPOLOGICAL SORT: Kahn's Algorithm (LeetCode 210)
# =====================================================================
def kahn_topological_sort_jsx(components: List[str], nesting_edges: List[Tuple[str, str]]) -> List[str]:
  """Orders nested React UI components in valid parent-child JSX rendering order.

  Fraz Module: Graphs TopoSort (LC 210)
  Time: O(V + E), Space: O(V + E)
  """
  in_degree = {c: 0 for c in components}
  adj = {c: [] for c in components}

  for parent, child in nesting_edges:
    if parent in adj and child in in_degree:
      adj[parent].append(child)
      in_degree[child] += 1

  queue = collections.deque([c for c, deg in in_degree.items() if deg == 0])
  order = []

  while queue:
    node = queue.popleft()
    order.append(node)
    for neighbor in adj[node]:
      in_degree[neighbor] -= 1
      if in_degree[neighbor] == 0:
        queue.append(neighbor)

  return order if len(order) == len(components) else components


# =====================================================================
# 13. GRAPH DIJKSTRA: Shortest Path Routing for UI Connectors (LeetCode 743)
# =====================================================================
def dijkstra_shortest_path(nodes: List[str], edges: List[Tuple[str, str, int]], start_node: str) -> Dict[str, int]:
  """Finds shortest layout connection distance between UI components via Dijkstra's algorithm.

  Fraz Module: Graphs Dijkstra (LC 743)
  Time: O((V + E) log V), Space: O(V)
  """
  adj = collections.defaultdict(list)
  for u, v, w in edges:
    adj[u].append((v, w))
    adj[v].append((u, w))

  dist = {node: float("inf") for node in nodes}
  dist[start_node] = 0
  min_heap = [(0, start_node)]

  while min_heap:
    d, u = heapq.heappop(min_heap)
    if d > dist[u]:
      continue
    for v, weight in adj[u]:
      if dist[u] + weight < dist[v]:
        dist[v] = dist[u] + weight
        heapq.heappush(min_heap, (dist[v], v))

  return dist


# =====================================================================
# 14. TREE LCA: Lowest Common Ancestor (LeetCode 236)
# =====================================================================
class TreeNode:
  def __init__(self, val: str):
    self.val = val
    self.left: Optional["TreeNode"] = None
    self.right: Optional["TreeNode"] = None


def lowest_common_ancestor_dom(root: Optional[TreeNode], p_val: str, q_val: str) -> Optional[TreeNode]:
  """Finds the shared parent container for two nested UI elements in the React DOM tree.

  Fraz Module: Trees / LCA (LC 236)
  Time: O(N), Space: O(H)
  """
  if not root or root.val == p_val or root.val == q_val:
    return root

  left = lowest_common_ancestor_dom(root.left, p_val, q_val)
  right = lowest_common_ancestor_dom(root.right, p_val, q_val)

  if left and right:
    return root
  return left if left else right


# =====================================================================
# 15. BINARY SEARCH ON ANSWER: Capacity to Fit Components (LeetCode 1011)
# =====================================================================
def binary_search_viewport_capacity(component_heights: List[int], max_pages: int = 2) -> int:
  """Binary search on answer for minimum viewport height required to fit UI components within max_pages.

  Fraz Module: Binary Search on Answer (LC 1011)
  Time: O(N log(Sum(H))), Space: O(1)
  """
  def can_fit(capacity):
    pages = 1
    curr = 0
    for h in component_heights:
      if curr + h > capacity:
        pages += 1
        curr = h
      else:
        curr += h
    return pages <= max_pages

  low = max(component_heights)
  high = sum(component_heights)
  ans = high

  while low <= high:
    mid = (low + high) // 2
    if can_fit(mid):
      ans = mid
      high = mid - 1
    else:
      low = mid + 1

  return ans


def binary_search_lower_bound(sorted_scores: List[float], min_confidence: float) -> int:
  """Binary search lower bound."""
  low, high = 0, len(sorted_scores)
  while low < high:
    mid = (low + high) // 2
    if sorted_scores[mid] >= min_confidence:
      high = mid
    else:
      low = mid + 1
  return low


# =====================================================================
# 16. TRIE: Prefix Tree for React Tags & Tailwind CSS (LeetCode 208)
# =====================================================================
class TrieNode:
  def __init__(self):
    self.children: Dict[str, "TrieNode"] = {}
    self.is_end_of_word: bool = False
    self.full_tag: str = ""


class ReactComponentTrie:
  """Trie data structure for fast prefix search of React component names and Tailwind classes."""

  def __init__(self):
    self.root = TrieNode()

  def insert(self, word: str):
    node = self.root
    for char in word:
      if char not in node.children:
        node.children[char] = TrieNode()
      node = node.children[char]
    node.is_end_of_word = True
    node.full_tag = word

  def autocomplete(self, prefix: str) -> List[str]:
    node = self.root
    for char in prefix:
      if char not in node.children:
        return []
      node = node.children[char]

    results = []
    def dfs(curr):
      if curr.is_end_of_word:
        results.append(curr.full_tag)
      for child in curr.children.values():
        dfs(child)

    dfs(node)
    return results


# =====================================================================
# 17. HEAPS: Top-K Frequent / Confident Elements (LeetCode 347)
# =====================================================================
def top_k_confident_detections(detections: List[Tuple[str, float]], k: int = 3) -> List[Tuple[str, float]]:
  """Extracts top-k most confident component predictions using a Min-Heap of size K."""
  min_heap = []
  for label, score in detections:
    if len(min_heap) < k:
      heapq.heappush(min_heap, (score, label))
    else:
      if score > min_heap[0][0]:
        heapq.heappop(min_heap)
        heapq.heappush(min_heap, (score, label))

  return sorted([(lbl, sc) for sc, lbl in min_heap], key=lambda x: x[1], reverse=True)


# =====================================================================
# 18. DYNAMIC PROGRAMMING: 0/1 Knapsack Viewport Budget Packing
# =====================================================================
def knapsack_viewport_packing(weights: List[int], values: List[int], max_capacity: int) -> int:
  """0/1 Knapsack DP for packing highest priority UI components into fixed viewport pixel budget.

  Fraz Module: Dynamic Programming (0/1 Knapsack)
  Time: O(N * W), Space: O(W)
  """
  dp = [0] * (max_capacity + 1)
  for w, val in zip(weights, values):
    for j in range(max_capacity, w - 1, -1):
      dp[j] = max(dp[j], dp[j - w] + val)
  return dp[max_capacity]


# =====================================================================
# 19. DYNAMIC PROGRAMMING: Edit Distance (LeetCode 72)
# =====================================================================
def layout_edit_distance(predicted_tags: List[str], template_tags: List[str]) -> int:
  """Computes minimum edit distance between predicted UI structure and design templates."""
  m, n = len(predicted_tags), len(template_tags)
  dp = [[0] * (n + 1) for _ in range(m + 1)]

  for i in range(m + 1):
    dp[i][0] = i
  for j in range(n + 1):
    dp[0][j] = j

  for i in range(1, m + 1):
    for j in range(1, n + 1):
      if predicted_tags[i - 1] == template_tags[j - 1]:
        dp[i][j] = dp[i - 1][j - 1]
      else:
        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

  return dp[m][n]


# =====================================================================
# 20. BIT MANIPULATION: Bitmask Style Flags
# =====================================================================
FLAG_BORDER   = 1 << 0  # 1
FLAG_SHADOW   = 1 << 1  # 2
FLAG_ROUNDED  = 1 << 2  # 4
FLAG_HOVER    = 1 << 3  # 8

def evaluate_bitmask_flags(mask: int) -> List[str]:
  """Extracts active CSS styling features using bit manipulation."""
  features = []
  if mask & FLAG_BORDER:
    features.append("border-2")
  if mask & FLAG_SHADOW:
    features.append("shadow-xl")
  if mask & FLAG_ROUNDED:
    features.append("rounded-2xl")
  if mask & FLAG_HOVER:
    features.append("hover:scale-105")
  return features
