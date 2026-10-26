# Sketch2React: Multimodal Wireframe-to-React AI Microservice

> **Deep Learning Course Project & Standalone AI Service**  
> Designed for seamless integration into **React-Comps** (Full-Stack Text/Prompt-to-React platform).  
> Covers **100% of Deep Learning Lab Curriculum (Labs 0–8)** + **Fraz's DSA Sheet Algorithmic Patterns**.

---

## 🎨 What Happens When You Draw on Canvas?

When you draw on the interactive canvas (`/demo`) or upload an image:
1. **Stroke Capture & Deskewing (`Lab 0`, `LC 304`)**: Pixels are deskewed via orthogonal transformation ($A \cdot A^T = I$). A 2D Prefix Sum matrix enables $O(1)$ density queries.
2. **Boundary Discovery (`LC 84`, `LC 200`)**: Monotonic stack finds maximal rectangular bounds; Graph BFS groups disconnected strokes into unified components.
3. **Deep Visual Classification (`Lab 7`)**: An 8-layer AlexNet with custom LRN, trained on a generated sketch dataset (`train.py`), classifies the sketch into one of 5 components (98.5% test accuracy).
4. **Layout Gate & Styling (`Lab 4`, `Lab 5`)**: A Perceptron ($w \cdot x \ge -w_0$) separates Button vs Card in the rule-based fallback. The style variant (Primary, Outline, Glassmorphism) comes from geometry rules; the Lab 5 Dropout ANN is not trained yet.
5. **Contrast Invariance (`Lab 3`)**: Batch Normalization over the pixel batch makes light pencil and dark marker sketches look the same to AlexNet. The Lab 2 scratch MLP's forward pass is reported in the trace.
6. **JSX Hierarchy Resolution (`LC 210`)**: Kahn's Topological Sort orders the DOM containment tree (`PageWrapper > Card > Button`).
7. **Agentic Code Synthesis (`LangGraph`, `Trie`)**: LangGraph StateGraph synthesizes complete JSX code enriched with Tailwind CSS utilities and Lucide icons via Trie tag lookup.
8. **Live Preview (`/demo`)**: The UI JSON and generated LLM prompt are rendered live on the interactive canvas playground and ready for React-Comps export.


---

## Architecture Overview

```
 [Hand-Drawn Sketch / Wireframe]
               │
               ▼
   [Stage 1: Preprocessing & Straightening]  ──>  Lab 0 (Orthogonal rotation R: deskew up to ±10°)
               │                             ──>  Fraz DSA: 2D Prefix Sum (LC 304) & Two Pointers (bounding box)
               ▼
   [Stage 2: Contrast Normalization]         ──>  Lab 3 (Batch Normalization over the pixel batch)
               │
               ▼
   [Stage 3: Component Classification]       ──>  Lab 7 (Trained AlexNet with Custom LRN): the prediction
               │                             ──>  Fraz DSA: Min-Heap Top-3 (LC 347) & Binary Search
               ▼
   [Stage 4: Style Variant]                  ──>  Geometry/ink-density rules (+ Lab 5 Dropout ANN, untrained)
               │
               ▼
   [Stage 5: Layout Analysis]                ──>  Fraz DSA: BFS Islands (LC 200), Monotonic Stack (LC 84), Merge Intervals (LC 56)
               │                             ──>  Lab 4 Perceptron gate (decides only in the no-weights fallback)
               ▼
   [Stage 6: JSX Hierarchy & Prompt]         ──>  Fraz DSA: Kahn's TopoSort (LC 210), Tag Matcher (LC 20), Trie
               │                             ──>  Lab 1 tensor ops & Lab 2 scratch MLP (reported in trace)
               ▼
 [Structured UI JSON + Enriched Prompt] ──> [React-Comps Vite Live Preview]

 Standalone revision modules (not in the pipeline): Lab 6 (GradientTape loop), Lab 8 (autoencoder pre-training)
```

---

## Laboratory Mapping & Quick Revision Guide

Each lab in the curriculum has a dedicated, standalone executable file in `core/` for exam and viva preparation. The detected component comes from **Lab 7** (AlexNet), fed by **Lab 0** (deskew) and **Lab 3** (normalization):

| Lab | Module File | Key Theory & Equations | Where it is used (✅ affects prediction, ⚠️ fallback only, ➖ trace/standalone) |
|---|---|---|---|
| **Lab 0** | `core/lab0_linear_algebra.py` | Orthogonal matrix ($A \cdot A^T = I$), Trace ($\sum A_{ii}$), Gaussian Elimination Upper Triangular ($\det = \prod U_{ii}$) | ✅ `pipeline.deskew()` measures the slant and straightens the sketch with $R(	heta)$ before classification. |
| **Lab 1** | `core/lab1_tf_graph.py` | `tf.constant`, tensor arithmetic (`tf.add`, `tf.subtract`), Keras `Sequential` graph execution | ➖ `tensor_arithmetic_demo()` value shown in the trace; all models are Keras graphs. |
| **Lab 2** | `core/lab2_scratch_backprop.py` | Sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$, derivative $\sigma'(z) = z(1-z)$, output $\delta^{[L]} = (y-\hat{y})\sigma'$, hidden $\delta^{[l]} = (\sum W^T\delta)\sigma'$, manual updates | ➖ `ScratchMLP.forward()` on geometric features; untrained, trace only. |
| **Lab 3** | `core/lab3_batch_norm.py` | $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$, $y = \gamma \hat{x} + \beta$. Scratch NumPy vs Keras `BatchNormalization` | ✅ `batch_norm_scratch()` in `preprocess_sketch_for_alexnet()` (training + inference): pen-contrast invariance. |
| **Lab 4** | `core/lab4_perceptron_boundary.py` | Perceptron decision rule: $w_1 x_1 + w_2 x_2 \ge -w_0$. Linear bounds: $\max_{y=0}(w \cdot x) < -w_0 \le \min_{y=1}(w \cdot x)$ | ⚠️ `classify_button_vs_card()`: used only by the rule-based fallback when no trained weights exist. |
| **Lab 5** | `core/lab5_style_ann.py` | Flatten $\to$ Dense(16, sigmoid) $\to$ Dropout(0.2) $\to$ Dense(8, sigmoid) $\to$ Dropout(0.2) $\to$ Softmax | ➖ `predict_component_style()`: untrained (no style labels yet); style variant comes from rules. |
| **Lab 6** | `core/lab6_cnn_custom_training.py` | Custom mini-batch loop: `np.random.permutation`, `tf.GradientTape()`, `tape.gradient()`, `optimizer.apply_gradients()` | ➖ Standalone module, not part of the pipeline. |
| **Lab 7** | `core/lab7_alexnet_backbone.py` | 8-layer AlexNet: $11 \times 11$ conv (stride 4), Custom LRN Layer (`tf.nn.local_response_normalization`), overlapping pool ($3 \times 3$, stride 2), Dropout 0.5 | ✅ **The classifier**: trained by `train.py`, loaded by `load_trained_alexnet()` (98.5% test accuracy). |
| **Lab 8** | `core/lab8_autoencoder_pretraining.py` | Overcoming vanishing gradients: random init gets stuck at 11% accuracy; greedy layer-wise Autoencoder MSE pre-training yields >96% accuracy | ➖ Standalone module, not part of the pipeline. |

---

## Fraz's DSA Sheet Algorithms Implemented (`core/fraz_algorithms.py`)

1. **2D Prefix Sum (LeetCode 304)**: $O(1)$ query time for bounding box stroke density.
2. **Two Pointers**: Converging pointers from left and right to detect tight component boundaries in $O(N)$.
3. **Monotonic Stack (LeetCode 84 - Largest Rectangle in Histogram)**: Finds maximal layout containers and card bounds in $O(N)$.
4. **Merge Intervals (LeetCode 56)**: Non-Maximum Suppression (NMS) merging overlapping bounding box proposals in $O(N \log N)$.
5. **Graph BFS (LeetCode 200 - Number of Islands)**: Groups disconnected pen strokes into unified component elements.
6. **Kahn's Topological Sort (LeetCode 210)**: Computes valid JSX parent-child rendering order (`PageWrapper > Card > Button`).
7. **Binary Search**: Lower bound search for dynamic confidence thresholding in $O(\log N)$.
8. **Trie (Prefix Tree)**: Instant prefix auto-completion for React components and Tailwind CSS utility classes in $O(L)$.
9. **Heaps / Priority Queue (LeetCode 347 - Top K Frequent)**: Min-heap of size $K$ extracting top-$k$ most confident predictions in $O(N \log K)$.
10. **Dynamic Programming (LeetCode 72 - Edit Distance)**: Levenshtein distance aligning detected component layouts with standard design system templates.

---

## Running the Service

### 1. Revise Any Specific Lab
Run any individual lab module directly to test its isolated logic:
```bash
python core/lab0_linear_algebra.py
python core/lab1_tf_graph.py
python core/lab2_scratch_backprop.py
python core/lab3_batch_norm.py
python core/lab4_perceptron_boundary.py
python core/lab5_style_ann.py
python core/lab6_cnn_custom_training.py
python core/lab7_alexnet_backbone.py
python core/lab8_autoencoder_pretraining.py
python core/fraz_algorithms.py
```
### 2. Start the FastAPI Microservice
Launch the server on port 8000:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
- **Interactive Canvas Drawing Playground**: **`http://localhost:8000/demo`**
- **Interactive Swagger API Documentation**: **`http://localhost:8000/docs`**

---

## How to Integrate into Your Project

`Sketch2React` can be integrated into existing web apps, microservices, or Python pipelines in four different ways:

---

### 1. Direct React Frontend Integration (e.g. In `React-Comps`)

Add an image upload or HTML5 Canvas drawing pad where users can draw wireframes. The client sends the sketch directly to the microservice:

```javascript
// In your React Component (e.g., SketchUploader.jsx)
import React, { useState } from 'react';

export function SketchUploader({ onPromptGenerated }) {
  const [loading, setLoading] = useState(false);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/predict-sketch', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      console.log('Detected Component:', data.detected_component);
      console.log('Style Variant:', data.style_variant);
      console.log('Suggested Prompt:', data.suggested_llm_prompt);

      // Pass the generated prompt directly to your existing React-Comps LLM generator!
      onPromptGenerated(data.suggested_llm_prompt);
    } catch (err) {
      console.error('Inference error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 border-2 border-dashed rounded-xl text-center">
      <input type="file" accept="image/*" onChange={handleFileUpload} />
      {loading && <p className="text-blue-500 mt-2">Analyzing sketch with AlexNet & LangGraph...</p>}
    </div>
  );
}
```

---

### 2. Node.js / Express Backend Proxy with Redis Caching

If you want your Node.js backend to manage authentication, rate limiting, and Redis caching before calling the Python microservice:

```javascript
// Express.js Route (routes/sketch.js)
const express = require('express');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
const Redis = require('ioredis');

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });
const redis = new Redis(); // Connect to Redis

router.post('/api/parse-wireframe', upload.single('sketch'), async (req, res) => {
  try {
    const fileHash = req.file.buffer.toString('base64').slice(0, 40); // Unique sketch hash
    const cachedResult = await redis.get(`sketch:${fileHash}`);

    if (cachedResult) {
      return res.json({ source: 'redis-cache', ...JSON.parse(cachedResult) });
    }

    // Forward image to Sketch2React Python microservice
    const formData = new FormData();
    formData.append('file', req.file.buffer, { filename: req.file.originalname });

    const aiResponse = await axios.post('http://localhost:8000/predict-sketch', formData, {
      headers: formData.getHeaders(),
    });

    // Cache prediction in Redis with 1-hour TTL
    await redis.set(`sketch:${fileHash}`, JSON.stringify(aiResponse.data), 'EX', 3600);

    return res.json({ source: 'sketch2react-service', ...aiResponse.data });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
});

module.exports = router;
```

---

### 3. Direct Python Module Import (No HTTP Server Needed)

If you have a Python script or AI pipeline and want to run inference in-process without spinning up FastAPI:

```python
import cv2
from pipeline import get_pipeline

# 1. Initialize the master pipeline
pipeline = get_pipeline()

# 2. Process any sketch (file path, numpy array, or base64)
result = pipeline.process("sample_sketches/card_sketch.png")

print("Component:", result["detected_component"])
print("Confidence:", result["confidence"])
print("Suggested LLM Prompt:\n", result["suggested_llm_prompt"])
```

---

### 4. Running the LangGraph Agentic Workflow Standalone

If you want to use the LangGraph state machine with custom agent nodes:

```python
import cv2
from core.langgraph_workflow import run_sketch_agent

# Read sketch image
image = cv2.imread("sample_sketches/button_sketch.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Execute the LangGraph StateGraph agent
final_state = run_sketch_agent(image_rgb)

print("Detected Component :", final_state["detected_component"])
print("Style Theme        :", final_state["style_variant"])
print("JSX DOM Hierarchy  :", " > ".join(final_state["jsx_hierarchy"]))
print("LangChain Prompt   :\n", final_state["enriched_prompt"])
```

