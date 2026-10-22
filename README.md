# DL_ProjectK: Deep Learning Laboratory Curriculum & Sketch2React AI Microservice

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the complete **Deep Learning Laboratory coursework (Labs 0–8)** and the **`Sketch2React` AI Microservice** designed to integrate multimodal vision capabilities into **[React-Comps](https://github.com/VeratrilS)**.

---

## 🎨 What Happens When You Draw on the Canvas? (The Complete Journey)

When a user draws a UI wireframe on the HTML5 canvas (`/demo`) or uploads an image, the system executes an automated, 10-stage pipeline converting raw pen strokes into production-ready React JSX:

```
[User Hand-Drawn Canvas Stroke]
               │
               ▼
   1. Canvas Stroke Capture & Base64 Transport (Browser HTML5 Canvas -> FastAPI)
               │
               ▼
   2. Stroke Deskewing & Density Grid       ──> Lab 0 (Orthogonal Matrix Rotation) + DSA (2D Prefix Sum LC 304)
               │
               ▼
   3. Dual-Axis Bounding Box & BFS Islands   ──> DSA (Two Pointers [L,R,T,B] + BFS Islands LC 200 + Monotonic Stack LC 84)
               │
               ▼
   4. Geometric Hyperplane Feasibility Gate ──> Lab 4 (Perceptron -w0 Threshold: Aspect Ratio vs Area)
               │
               ▼
   5. Pen Contrast Normalization            ──> Lab 3 (Batch Normalization across Pen Pressures)
               │
               ▼
   6. Deep Visual Backbone Inference        ──> Lab 7 (AlexNet with Custom LRN) + Lab 8 (Stacked Autoencoder)
               │
               ▼
   7. Analytical Confidence Calibration     ──> Lab 2 (First-Principles Analytical Backpropagation)
               │
               ▼
   8. Component Variant & Styling ANN       ──> Lab 5 (Multi-Layer ANN with Dropout 0.2)
               │
               ▼
   9. Spatial Graph & TopoSort Ordering     ──> DSA (Kahn's TopoSort LC 210: Parent > Child JSX Tree)
               │
               ▼
  10. Agentic Code Synthesis & Live Preview ──> LangGraph StateGraph + Trie (Tailwind) + Interactive /demo
```

### Step-by-Step Data Flow Breakdown:

1. **Canvas Event & Stroke Serialization (Browser)**:
   The user draws with mouse/touch on a 227x227 HTML5 Canvas (`#1e293b` strokes on `#f8fafc` background). Clicking **"Analyze Sketch"** calls `canvas.toDataURL('image/png')` which serializes the drawing into a Base64 PNG string (`data:image/png;base64,...`) and sends an HTTP POST request to `/predict-sketch`.
2. **Deskewing & Linear Algebra Alignment (`Lab 0` & `LeetCode 304`)**:
   FastAPI ingests the image and converts it into a `(227, 227, 3)` RGB float tensor and a normalized luminance matrix `raw_gray` $\in [0, 1]$.
   - *Lab 0 (Orthogonal Matrix)*: Straightens slanted strokes using an orthogonal rotation matrix $R(\theta)$ ($R \cdot R^T = I$, $\det(R)=1$).
   - *2D Prefix Sum (LC 304)*: Computes a 2D cumulative sum grid $P[r][c] = \sum$ for $O(1)$ sub-region ink density queries.
3. **Dual-Axis Bounding Box & Islands (`LeetCode 200`, `Two Pointers`, `LC 84`)**:
   - *Two Pointers*: Converges horizontally on column ink projection `h_proj` to find $[L, R]$, and vertically on row ink projection `v_proj` to find $[T, B]$:
     $\text{Width} = R - L$, $\text{Height} = B - T$, $\text{Aspect Ratio} = \text{Width} / \text{Height}$, $\text{Area} = (\text{Width} \cdot \text{Height}) / 227^2$, $Y_{\text{center}} = (T + B) / (2 \cdot 227)$.
   - *BFS Islands (LC 200)*: Flood-fills connected black pixels to count discrete UI elements (e.g., a Card with header, text, and button has 26 stroke islands, while a simple Button has 6).
   - *Monotonic Stack (LC 84)*: Finds the maximal bounding rectangle inside the stroke histogram.
4. **Perceptron Hyperplane Gate (`Lab 4`)**:
   Evaluates geometric linear separability using the Lab 4 Perceptron rule:
   $$\text{Score} = w_1 \cdot \text{Aspect Ratio} + w_2 \cdot \text{Normalized Area} - w_0$$
   Separates wide/compact elements ($+1 \to \text{Button}$) from tall/large elements ($-1 \to \text{Card}$).
5. **Batch Normalization for Contrast Invariance (`Lab 3`)**:
   Applies zero-mean, unit-variance normalization:
   $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
   Standardizes light pencil strokes, dark ink pens, and varied background lighting so neural activations do not saturate.
6. **Deep Visual Feature Extraction (`Lab 7` & `Lab 8`)**:
   - *Lab 7 (AlexNet Backbone)*: Processes the normalized image through 5 Convolution layers with custom **Local Response Normalization (LRN)** and overlapping max pooling.
   - *Lab 8 (Stacked Autoencoder)*: Reconstructs latent representations to ensure robust cold-start representations.
7. **Analytical Confidence Calibration (`Lab 2`)**:
   Inputs $[\text{Aspect Ratio}, \text{Normalized Area}, \text{Base Confidence}]$ are forwarded through a 3-layer scratch MLP. Analytical Sigmoid gradients $\sigma'(z) = z(1-z)$ calibrate output softmax probabilities, preventing overconfident hallucinations.
8. **Component Styling & Variant Prediction (`Lab 5`)**:
   A 28x28 component patch is passed to the Lab 5 ANN regularized with **Dropout (p = 0.2)**. Combined with ink density ($1.0 - \text{density}$), it classifies styling into:
   - `Elevated Modern Card` (for cards with internal containers)
   - `Glassmorphism / Shadow` (for centered dialog modals)
   - `Outline Bordered` (for clean input fields & elongated borders)
   - `Primary Filled` (for solid, dense clickable buttons)
   - `Subtle Minimal` (for clean, minimalist elements)
9. **JSX Spatial Hierarchy & Topological Sorting (`LeetCode 210`)**:
   Constructs a directed acyclic containment graph. Kahn's Topological Sort computes valid parent-to-child nesting:
   $$\text{PageWrapper} \longrightarrow \text{Container} \longrightarrow \text{Card} \longrightarrow \text{Button}$$
   A Monotonic Stack (LeetCode 20) validates all opening and closing JSX tags.
10. **Agentic Code Synthesis & Live Playground Preview (`LangGraph` + Trie)**:
    - *Trie Prefix Tree*: Matches Tailwind CSS utility classes (`rounded-xl`, `shadow-lg`, `hover:bg-blue-600`) and Lucide React icons in $O(L)$ time.
    - *LangGraph StateGraph*: Synthesizes clean, accessible React code and returns structured JSON to the interactive `/demo` canvas playground and React-Comps LLM engine.

---

## 📂 Repository Structure

```
DL_ProjectK/
├── lab/                                    # Deep Learning Lab Notebooks & Assignments
│   ├── Lab0.ipynb                          # Lab 0: NumPy Linear Algebra, Orthogonality, Det, Trace
│   ├── Lab1.ipynb                          # Lab 1: TensorFlow Basics & Sequential Model
│   ├── Lab2.ipynb                          # Lab 2: Pure Math Backpropagation from Scratch
│   ├── Lab3.ipynb                          # Lab 3: Batch Normalization (NumPy Scratch vs Keras)
│   ├── Lab4.ipynb                          # Lab 4: Perceptron Logic Gates & -w0 Threshold
│   ├── Lab5.ipynb                          # Lab 5: Deep ANN with Dropout (0.2) on MNIST
│   ├── Lab6.ipynb                          # Lab 6: CNN with Custom tf.GradientTape Mini-Batch Loop
│   ├── Lab6_scratch.ipynb                  # Lab 6 Scratch: 2D Convolution & Backprop via SciPy
│   ├── Lab7/                               # Lab 7: AlexNet Architecture from Scratch
│   │   ├── alexnet_model.ipynb             # Custom LRN Layer, Overlapping Pooling, Full AlexNet
│   │   └── alexnet_report.pdf              # Comprehensive Analysis & Learning Curve Report
│   └── lab8/                               # Lab 8: Unsupervised Pre-training with Autoencoders
│       ├── Lab8_Unsupervised_Pretraining.ipynb # Greedy Layer-Wise Stacked Autoencoder Pre-training
│       └── 123cs0211_Lab8.pdf              # Vanishing Gradient Analysis Report
│
└── sketch2react_service/                   # Production AI Microservice for React-Comps
    ├── core/                               # Modular Lab Implementations & Fraz's DSA Suite
    │   ├── lab0_linear_algebra.py          # Lab 0: Orthogonal Matrices & Stroke Alignment
    │   ├── lab1_tf_graph.py                # Lab 1: Immutable Tensors & Graph Pipeline
    │   ├── lab2_scratch_backprop.py        # Lab 2: First-Principles Backprop Confidence Calibrator
    │   ├── lab3_batch_norm.py              # Lab 3: Batch Normalization for Pen Contrast Invariance
    │   ├── lab4_perceptron_boundary.py     # Lab 4: Perceptron Geometric Layout Classifier
    │   ├── lab5_style_ann.py               # Lab 5: Component Style & Variant Predictor (Dropout 0.2)
    │   ├── lab6_cnn_custom_training.py     # Lab 6: CNN + Custom tf.GradientTape Optimizer
    │   ├── lab7_alexnet_backbone.py        # Lab 7: AlexNet + Custom Local Response Normalization
    │   ├── lab8_autoencoder_pretraining.py # Lab 8: Stacked Autoencoder Representation Learning
    │   ├── fraz_algorithms.py              # Fraz DSA Suite (Prefix Sum, Stack, TopoSort, Trie, Heaps)
    │   └── langgraph_workflow.py           # LangGraph StateGraph Agentic Workflow
    ├── pipeline.py                         # Master End-to-End Vision-to-React Pipeline
    ├── api.py                              # FastAPI REST Microservice with Live Canvas Demo
    ├── sample_sketches/                    # Realistic Hand-Drawn UI Sketches
    └── README.md                           # Microservice Architecture & Integration Guide
```

---

## 📚 Lab Curriculum Summary (Quick Brief)

A concise reference table summarizing the core theoretical concepts and their production roles in `Sketch2React`:

| Lab | Core Topic | Key Equation / Technique | Role in Sketch2React |
|---|---|---|---|
| **Lab 0** | NumPy Linear Algebra | $A \cdot A^T = I$, $\det = \prod U_{ii}$, $\text{Tr}(A) = \sum A_{ii}$ | Stroke deskewing via orthogonal transformation |
| **Lab 1** | TensorFlow Graphs | `tf.constant`, tensor arithmetic, `Sequential` | Immutable tensor representation & graph compilation |
| **Lab 2** | Pure Math Backpropagation | Sigmoid $\sigma' = z(1-z)$, analytical $\delta^{[l]}$ updates | Confidence calibration against gradient drift |
| **Lab 3** | Batch Normalization | $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$, $y_i = \gamma \hat{x}_i + \beta$ | Stroke & paper contrast invariance across pens |
| **Lab 4** | Perceptron & Boundaries | $w_1 x_1 + w_2 x_2 \ge -w_0$, linear feasibility bounds | Fast geometric classification (Button vs Card) |
| **Lab 5** | Deep ANN with Dropout | Dropout(0.2) regularization, Softmax multi-class | Component variant prediction (Primary, Glass, etc.) |
| **Lab 6** | CNN & Custom Training Loop | `tf.GradientTape()`, manual mini-batch permutation | Contour & edge feature representation learning |
| **Lab 7** | AlexNet Backbone | $11 \times 11$ Conv, Custom LRN Layer, Overlapping Pool | Primary visual backbone classifying 5 UI primitives |
| **Lab 8** | Autoencoder Pre-training | Greedy layer-wise MSE reconstruction (>96% acc) | Pre-training representations on unlabeled sketches |

---

## 💡 Fraz's DSA Algorithmic Suite (`core/fraz_algorithms.py`)

1. **2D Prefix Sum (LC 304)**: $O(1)$ query time for bounding box stroke density.
2. **Two Pointers**: Converging horizontal/vertical projections for boundary discovery in $O(N)$.
3. **Monotonic Stack (LC 84)**: Maximal container area detection in histogram representation in $O(N)$.
4. **Merge Intervals (LC 56)**: Non-Maximum Suppression (NMS) merging overlapping bounding box proposals in $O(N \log N)$.
5. **Graph BFS (LC 200)**: Connected component labeling grouping isolated pen marks into unified UI widgets.
6. **Kahn's Topological Sort (LC 210)**: Computes valid JSX nesting tree (`PageWrapper > Card > Button`).
7. **Binary Search**: Dynamic confidence threshold filtering in $O(\log N)$.
8. **Trie (Prefix Tree)**: $O(L)$ tag and class name autocompletion for React & Tailwind CSS.
9. **Priority Queue / Min-Heap (LC 347)**: Top-$k$ high-confidence proposal extraction in $O(N \log K)$.
10. **Dynamic Programming (LC 72)**: Levenshtein distance matching detected components to design system schemas.

---

## 🚀 Quickstart

### 1. Launch FastAPI Server & Live Canvas
```bash
cd sketch2react_service
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

- **Interactive Canvas Playground**: [`http://127.0.0.1:8000/demo`](http://127.0.0.1:8000/demo)
- **Swagger REST API Documentation**: [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)

---

## 👤 Author

**Nandam Venkata Sai Raghava**  
[GitHub Profile](https://github.com/VeratrilS)
