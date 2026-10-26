# DL_ProjectK: Deep Learning Laboratory Curriculum & Sketch2React AI Microservice

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the complete **Deep Learning Laboratory coursework (Labs 0–8)** and the **`Sketch2React` AI Microservice** designed to integrate multimodal vision capabilities into **[React-Comps](https://github.com/VeratrilS)**.

**In one paragraph:** you draw a UI element (Button, Card, InputField, Navbar or Modal) on a canvas or upload a photo of a sketch. The image is straightened (Lab 0), normalized (Lab 3) and classified by an AlexNet CNN (Lab 7) trained on 2,000 generated sketches (98.5% test accuracy). The service returns the detected component, a confidence score, a style variant, a ready-to-use React + Tailwind template for that component, and an enriched prompt that React-Comps' LLM can use to generate a custom version. Every response also includes a trace showing the other labs and the DSA algorithms that ran on the sketch.

---

## 🎨 What Happens When You Draw on the Canvas? (The Complete Journey)

When a user draws a UI wireframe on the HTML5 canvas (`/demo`) or uploads an image, the system runs the following pipeline (implemented in `sketch2react_service/pipeline.py`):

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
   4. Geometric Hyperplane Gate             ──> Lab 4 (Perceptron: decides only in the no-weights fallback)
               │
               ▼
   5. Pen Contrast Normalization            ──> Lab 3 (Batch Normalization across Pen Pressures)
               │
               ▼
   6. Deep Visual Backbone Inference        ──> Lab 7 (Trained AlexNet with Custom LRN) — the classifier
               │
               ▼
   7. Scratch MLP Trace                     ──> Lab 2 (Scratch MLP forward pass, trace only)
               │
               ▼
   8. Component Variant & Styling ANN       ──> Lab 5 (Dropout ANN, untrained) + style rules
               │
               ▼
   9. Spatial Graph & TopoSort Ordering     ──> DSA (Kahn's TopoSort LC 210: Parent > Child JSX Tree)
               │
               ▼
  10. React Template + LLM Prompt           ──> React/Tailwind template per component + Trie + /demo preview
```

### Step-by-Step Data Flow Breakdown:

1. **Canvas Event & Stroke Serialization (Browser)**:
   The user draws with mouse/touch on a 227x227 HTML5 Canvas (`#0f172a` strokes on a white background). Clicking **"Convert to React"** calls `canvas.toDataURL('image/png')` which serializes the drawing into a Base64 PNG string (`data:image/png;base64,...`) and sends an HTTP POST request to `/predict-sketch`.
2. **Deskewing & Linear Algebra Alignment (`Lab 0` & `LeetCode 304`)**:
   FastAPI ingests the image and converts it into a 227x227 grayscale image and an ink map $\in [0, 1]$ (1 = stroke, 0 = paper).
   - *Lab 0 (Orthogonal Matrix)*: Estimates the slant from the ink's minimum-area rectangle and straightens it (up to ±10°) using an orthogonal rotation matrix $R(\theta)$ ($R \cdot R^T = I$, $\det(R)=1$).
   - *2D Prefix Sum (LC 304)*: Computes a 2D cumulative sum grid $P[r][c] = \sum$ for $O(1)$ sub-region ink density queries.
3. **Dual-Axis Bounding Box & Islands (`LeetCode 200`, `Two Pointers`, `LC 84`)**:
   - *Two Pointers*: Converges horizontally on column ink projection `h_proj` to find $[L, R]$, and vertically on row ink projection `v_proj` to find $[T, B]$:
     $\text{Width} = R - L$, $\text{Height} = B - T$, $\text{Aspect Ratio} = \text{Width} / \text{Height}$, $\text{Area} = (\text{Width} \cdot \text{Height}) / 227^2$, $Y_{\text{center}} = (T + B) / (2 \cdot 227)$.
   - *BFS Islands (LC 200)*: Flood-fills connected black pixels on a 28x28 ink patch to count discrete stroke groups (a Card with image, text and button typically has 10+ islands; a plain Button 1-2).
   - *Monotonic Stack (LC 84)*: Finds the maximal bounding rectangle inside the stroke histogram.
4. **Perceptron Hyperplane Gate (`Lab 4`)**:
   Evaluates geometric linear separability using the Lab 4 Perceptron rule:
   $$\text{Score} = w_1 \cdot \text{Aspect Ratio} + w_2 \cdot \text{Normalized Area} - w_0$$
   Separates wide/compact elements ($+1 \to \text{Button}$) from tall/large elements ($-1 \to \text{Card}$). With trained AlexNet weights this decision is only reported in the trace; it is used by the rule-based fallback when no weights exist.
5. **Batch Normalization for Contrast Invariance (`Lab 3`)**:
   Applies zero-mean, unit-variance normalization:
   $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
   Standardizes light pencil strokes, dark ink pens, and varied background lighting so neural activations do not saturate.
6. **Deep Visual Classification (`Lab 7`)**:
   - *Lab 7 (AlexNet Backbone)*: Processes the normalized image through 5 Convolution layers with custom **Local Response Normalization (LRN)** and overlapping max pooling.
   - AlexNet is trained on a generated dataset of hand-drawn sketches (see **Training the Model** below); its softmax output is the detected component and confidence.
   - *Lab 6 / Lab 8* (custom GradientTape loop, autoencoder pre-training) are standalone modules in `core/` and are not part of inference.
7. **Scratch MLP Forward Pass (`Lab 2`)**:
   Inputs $[\text{Aspect Ratio}, \text{Normalized Area}, \text{Base Confidence}]$ are forwarded through a 3-layer scratch MLP. Analytical Sigmoid gradients $\sigma'(z) = z(1-z)$ are implemented in `train_step`; at inference the (untrained) MLP's forward pass is reported in the trace only.
8. **Component Styling & Variant Prediction (`Lab 5`)**:
   A 28x28 component patch is passed to the Lab 5 ANN regularized with **Dropout (p = 0.2)** (untrained: there are no style labels yet). The variant is chosen from the component and ink density:
   - `Elevated Modern Card` (for cards with internal containers)
   - `Glassmorphism / Shadow` (for centered dialog modals)
   - `Outline Bordered` (for clean input fields & elongated borders)
   - `Primary Filled` (for solid, dense clickable buttons)
   - `Subtle Minimal` (for clean, minimalist elements)
9. **JSX Spatial Hierarchy & Topological Sorting (`LeetCode 210`)**:
   Constructs a directed acyclic containment graph. Kahn's Topological Sort computes valid parent-to-child nesting:
   $$\text{PageWrapper} \longrightarrow \text{Container} \longrightarrow \text{Card} \longrightarrow \text{Button}$$
   A Monotonic Stack (LeetCode 20) validates all opening and closing JSX tags.
10. **React Template, LLM Prompt & Live Preview (Trie)**:
    - *React template*: each of the 5 components has a hand-written, accessible React + Tailwind template (`REACT_TEMPLATES` in `pipeline.py`); the detected component's template is returned as `generated_react_code`.
    - *LLM prompt*: `suggested_llm_prompt` describes the component, style variant and JSX hierarchy so React-Comps' LLM can generate a customised version.
    - *Trie Prefix Tree*: autocompletes Tailwind utility classes (`rounded-xl`, `hover:bg-blue-600`) in $O(L)$ time.
    - *LangGraph*: `core/langgraph_workflow.py` runs the same steps as a LangGraph StateGraph with a LangChain `PromptTemplate`. It is a standalone alternative entry point (`run_sketch_agent()`); the REST API uses `pipeline.py`.

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
    ├── generate_dataset.py                 # Generates the labelled sketch dataset (train/val/test)
    ├── train.py                            # Trains the Lab 7 AlexNet classifier
    ├── evaluate.py                         # End-to-end accuracy + confusion matrix on the test split
    ├── generate_sketches.py                # Regenerates the 5 demo sketches in sample_sketches/
    ├── models/alexnet_sketch.weights.h5    # Trained AlexNet weights (created by train.py)
    ├── dataset/                            # Generated dataset (git-ignored; created by generate_dataset.py)
    ├── sample_sketches/                    # Realistic Hand-Drawn UI Sketches
    └── README.md                           # Microservice Architecture & Integration Guide
```

---

## 📚 Lab Curriculum Summary (Quick Brief)

Which lab assignment is used where, and whether it actually affects the predicted component:

| Lab | Core Topic | Key Equation / Technique | Where it is used in the code | Affects the prediction? |
|---|---|---|---|---|
| **Lab 0** | NumPy Linear Algebra | $A \cdot A^T = I$, $\det = \prod U_{ii}$, $\text{Tr}(A) = \sum A_{ii}$ | `pipeline.deskew()`: measures the sketch's slant and straightens it with the orthogonal rotation matrix $R(\theta)$; det and trace of $R$ are checked | ✅ Yes: the image is straightened before classification |
| **Lab 1** | TensorFlow Graphs | `tf.constant`, tensor arithmetic, `Sequential` | `tensor_arithmetic_demo()` called in `pipeline.process()`; every model is a Keras graph | ➖ No: value shown in the algorithm trace only |
| **Lab 2** | Pure Math Backpropagation | Sigmoid $\sigma' = z(1-z)$, analytical $\delta^{[l]}$ updates | `ScratchMLP.forward()` on the sketch's geometric features in `pipeline.process()` | ➖ No: untrained, output shown in the trace only |
| **Lab 3** | Batch Normalization | $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$, $y_i = \gamma \hat{x}_i + \beta$ | `batch_norm_scratch()` inside `preprocess_sketch_for_alexnet()`, used by both `train.py` and the live pipeline | ✅ Yes: a light pencil and a dark marker sketch look the same to AlexNet |
| **Lab 4** | Perceptron & Boundaries | $w_1 x_1 + w_2 x_2 \ge -w_0$, linear feasibility bounds | `classify_button_vs_card()` in `pipeline.process()` | ⚠️ Only in the rule-based fallback (used when no trained weights exist) |
| **Lab 5** | Deep ANN with Dropout | Dropout(0.2) regularization, Softmax multi-class | `predict_component_style()` in `pipeline.process()` | ➖ No: untrained (no style-labelled data yet); the style variant comes from rules |
| **Lab 6** | CNN & Custom Training Loop | `tf.GradientTape()`, manual mini-batch permutation | `core/lab6_cnn_custom_training.py` (standalone, run it directly) | ➖ No: not part of the pipeline |
| **Lab 7** | AlexNet Backbone | $11 \times 11$ Conv, Custom LRN Layer, Overlapping Pool | `build_alexnet_model()` trained by `train.py`, loaded by `load_trained_alexnet()`, run in `Sketch2ReactPipeline.classify()` | ✅ **Yes: this is the classifier (98.5% test accuracy)** |
| **Lab 8** | Autoencoder Pre-training | Greedy layer-wise MSE reconstruction (>96% acc) | `core/lab8_autoencoder_pretraining.py` (standalone, run it directly) | ➖ No: not part of the pipeline |

**In short:** the detected component comes from **Lab 7** (AlexNet), fed by **Lab 0** (deskew) and **Lab 3** (normalization). The other labs are runnable revision modules, and their outputs appear in the `all_algorithms_executed` trace of every API response.

**Supporting scripts** (not tied to a single lab): `generate_dataset.py` builds the labelled sketch dataset, `train.py` trains the Lab 7 AlexNet (Adam, Dropout 0.5, early stopping on validation accuracy), and `evaluate.py` measures end-to-end accuracy with a confusion matrix.

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

## 🏋️ Training the Model

The AlexNet backbone is trained on a generated dataset of hand-drawn UI sketches (random size, position, pen thickness/darkness, stroke jitter, rotation, paper noise and class-specific inner details).

```bash
cd sketch2react_service
python generate_dataset.py                 # dataset/{train,val,test}/<Class>/*.png  (800/150/150 per class, seeded)
python train.py --max-per-class 400        # saves models/alexnet_sketch.weights.h5 (~70 min on CPU)
python evaluate.py                         # end-to-end pipeline accuracy on dataset/test
```

Results on the 750-image held-out test split:

| Classifier | Test accuracy | Weakest class |
|---|---|---|
| Old geometric rules (no trained model) | 74.1% | Modal (36.7% recall) |
| Trained AlexNet, full pipeline | **98.5%** | Button (94.0% recall, wide buttons read as InputField) |

Note: the test set is generated with the same procedure as the training set, so real hand-drawn sketches will score lower. Without trained weights the pipeline falls back to the geometric rules.

---

## 🚀 Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Make sure the trained model exists
The server loads `sketch2react_service/models/alexnet_sketch.weights.h5`. If it is missing, run the three commands in **Training the Model** above; otherwise the service falls back to the 74%-accurate geometric rules (it prints a warning at startup).

### 3. Launch FastAPI Server & Live Canvas
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
