# Sketch2React: Multimodal Wireframe-to-React AI Microservice

> **Deep Learning Course Project & Standalone AI Service**  
> Designed for seamless integration into **React-Comps** (Full-Stack Text/Prompt-to-React platform).  
> Covers **100% of Deep Learning Lab Curriculum (Labs 0–8)** + **Fraz's DSA Sheet Algorithmic Patterns**.

---

## 🎨 What Happens When You Draw on Canvas?

When you draw on the interactive canvas (`/demo`) or upload an image:
1. **Stroke Capture & Deskewing (`Lab 0`, `LC 304`)**: Pixels are deskewed via orthogonal transformation ($A \cdot A^T = I$). A 2D Prefix Sum matrix enables $O(1)$ density queries.
2. **Boundary Discovery (`LC 84`, `LC 200`)**: Monotonic stack finds maximal rectangular bounds; Graph BFS groups disconnected strokes into unified components.
3. **Deep Feature Extraction (`Lab 7`, `Lab 8`)**: Stacked Autoencoders (Lab 8) and an 8-layer AlexNet backbone with custom LRN (Lab 7) extract high-level visual representations.
4. **Layout Verification & Styling (`Lab 4`, `Lab 5`)**: A Perceptron ($w \cdot x \ge -w_0$) enforces geometric aspect ratios (Button vs Card). An ANN with 0.2 Dropout predicts style variants (Primary, Outline, Glassmorphism).
5. **Contrast Invariance & Calibration (`Lab 3`, `Lab 2`)**: Batch Normalization removes variations across pen types. An analytical backpropagation calibrator verifies confidence scores.
6. **JSX Hierarchy Resolution (`LC 210`)**: Kahn's Topological Sort orders the DOM containment tree (`PageWrapper > Card > Button`).
7. **Agentic Code Synthesis (`LangGraph`, `Trie`)**: LangGraph StateGraph synthesizes complete JSX code enriched with Tailwind CSS utilities and Lucide icons via Trie tag lookup.
8. **Live Preview (`/demo`)**: The UI JSON and generated LLM prompt are rendered live on the interactive canvas playground and ready for React-Comps export.


---

## Architecture Overview

```
 [Hand-Drawn Sketch / Wireframe]
               │
               ▼
   [Stage 1: Preprocessing & Straightening]  ──>  Lab 0 (Linear Algebra: Orthogonality & Trace)
               │                             ──>  Fraz DSA: 2D Prefix Sum (LC 304) & Two Pointers
               ▼
   [Stage 2: Visual Feature Extraction]      ──>  Lab 7 (AlexNet with Custom LRN Layer)
               │                             ──>  Lab 6 (2D Spatial Convolutions & MaxPool)
               ▼
   [Stage 3: Feature Stabilization]          ──>  Lab 3 (Batch Normalization across Pen Contrasts)
               │
               ▼
   [Stage 4: Latent Representation Pre-train]──>  Lab 8 (Stacked Autoencoder Pre-training)
               │
               ▼
   [Stage 5: Custom Mini-Batch Training]     ──>  Lab 6 (tf.GradientTape Custom Training Loop)
               │
               ▼
   [Stage 6: Style & Variant Classification] ──>  Lab 5 (Multi-Layer ANN with Dropout 0.2)
               │
               ▼
   [Stage 7: Layout Decision Boundary]       ──>  Lab 4 (Perceptron & -w0 Threshold Algorithm)
               │                             ──>  Fraz DSA: Monotonic Stack (LC 84)
               ▼
   [Stage 8: Component Ordering & NMS]       ──>  Fraz DSA: Kahn's TopoSort & Merge Intervals (LC 56)
               │                             ──>  Fraz DSA: Trie (Prefix Tree) for Tailwind/React Tags
               │                             ──>  Lab 2 (First-Principles Backpropagation Validation)
               ▼
 [Structured UI JSON + Enriched Prompt] ──> [React-Comps Vite Live Preview]
```

---

## Laboratory Mapping & Quick Revision Guide

Each lab in the curriculum has a dedicated, standalone executable file in `core/` for exam and viva preparation:

| Lab | Module File | Key Theory & Equations | Role in Sketch2React |
|---|---|---|---|
| **Lab 0** | `core/lab0_linear_algebra.py` | Orthogonal matrix ($A \cdot A^T = I$), Trace ($\sum A_{ii}$), Gaussian Elimination Upper Triangular ($\det = \prod U_{ii}$) | Straightening hand-drawn strokes with orthogonal rotation matrices and verifying zero shape distortion. |
| **Lab 1** | `core/lab1_tf_graph.py` | `tf.constant`, tensor arithmetic (`tf.add`, `tf.subtract`), Keras `Sequential` graph execution | Immutable image tensor conversions and model compilation. |
| **Lab 2** | `core/lab2_scratch_backprop.py` | Sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$, derivative $\sigma'(z) = z(1-z)$, output $\delta^{[L]} = (y-\hat{y})\sigma'$, hidden $\delta^{[l]} = (\sum W^T\delta)\sigma'$, manual updates | Pure mathematical backpropagation validating analytical gradients against automatic differentiation. |
| **Lab 3** | `core/lab3_batch_norm.py` | $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$, $y = \gamma \hat{x} + \beta$. Scratch NumPy vs Keras `BatchNormalization` | Normalizing stroke contrast across different pencils, pens, and paper brightness. |
| **Lab 4** | `core/lab4_perceptron_boundary.py` | Perceptron decision rule: $w_1 x_1 + w_2 x_2 \ge -w_0$. Linear bounds: $\max_{y=0}(w \cdot x) < -w_0 \le \min_{y=1}(w \cdot x)$ | Fast geometric layout boundary deciding between Button and Card based on aspect ratio and area. |
| **Lab 5** | `core/lab5_style_ann.py` | Flatten $\to$ Dense(16, sigmoid) $\to$ Dropout(0.2) $\to$ Dense(8, sigmoid) $\to$ Dropout(0.2) $\to$ Softmax | Predicting styling variants (Primary, Secondary, Outline, Danger) while preventing overfitting. |
| **Lab 6** | `core/lab6_cnn_custom_training.py` | Custom mini-batch loop: `np.random.permutation`, `tf.GradientTape()`, `tape.gradient()`, `optimizer.apply_gradients()` | Training edge and contour detection layers with custom class-weighted loss. |
| **Lab 7** | `core/lab7_alexnet_backbone.py` | 8-layer AlexNet: $11 \times 11$ conv (stride 4), Custom LRN Layer (`tf.nn.local_response_normalization`), overlapping pool ($3 \times 3$, stride 2), Dropout 0.5 | Primary visual backbone classifying sketches into 5 React component categories. |
| **Lab 8** | `core/lab8_autoencoder_pretraining.py` | Overcoming vanishing gradients: random init gets stuck at 11% accuracy; greedy layer-wise Autoencoder MSE pre-training yields >96% accuracy | Pre-training representations on thousands of unlabeled sketches before fine-tuning. |

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

