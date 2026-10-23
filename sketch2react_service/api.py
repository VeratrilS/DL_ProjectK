"""FastAPI Microservice for Sketch2React.

Exposes REST endpoints to be consumed by React-Comps (or tested independently):
- GET  /              -> Service status and overview
- POST /predict-sketch -> Accepts an image upload or base64 and returns component prediction + React-Comps prompt
- GET  /labs          -> Complete revision notes & syllabus guide for DL Labs 0 through 8
- GET  /fraz-algos    -> Summary of Fraz DSA Sheet algorithms used in the service
"""

import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from pipeline import UI_CLASSES, get_pipeline


@asynccontextmanager
async def lifespan(_app: FastAPI):
  get_pipeline()  # pre-warm the models before the first request
  yield


app = FastAPI(
    title="Sketch2React AI Microservice",
    description="Multimodal Hand-Drawn Sketch-to-React Component AI Engine (DL Labs 0-8 + Fraz DSA Sheet)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React-Comps frontend (Vite running on localhost:5173 or 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
  return {
      "service": "Sketch2React AI Engine",
      "status": "online",
      "architecture": "AlexNet (Lab 7) + Stacked Autoencoder (Lab 8) + Fraz DSA Suite",
      "integrated_labs": ["Lab 0", "Lab 1", "Lab 2", "Lab 3", "Lab 4", "Lab 5", "Lab 6", "Lab 7", "Lab 8"],
      "supported_components": UI_CLASSES,
      "interactive_demo": "/demo",
      "docs_url": "/docs",
  }


@app.get("/demo", response_class=HTMLResponse)
def interactive_demo_page():
  """Serves a clean, streamlined Input-to-Output playground for Sketch2React."""
  return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sketch2React Playground</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #f8fafc; margin: 0; padding: 32px 20px; }
    .container { max-width: 960px; margin: 0 auto; }
    .header { text-align: center; margin-bottom: 28px; }
    h1 { margin: 0 0 6px 0; font-size: 26px; color: #38bdf8; letter-spacing: -0.5px; }
    p.sub { color: #94a3b8; margin: 0; font-size: 14px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    .panel { background: #131b2e; padding: 24px; border-radius: 14px; border: 1px solid #1e293b; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
    .panel-title { font-size: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; margin-top: 0; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
    canvas { background: #ffffff; border-radius: 10px; cursor: crosshair; display: block; margin: 0 auto 16px auto; border: 2px solid #334155; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05); }
    .btn-row { display: flex; gap: 10px; justify-content: center; }
    button { background: #0284c7; color: white; border: none; padding: 10px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 14px; transition: all 0.15s ease; }
    button:hover { background: #0369a1; }
    button.secondary { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; }
    button.secondary:hover { background: #334155; color: white; }
    .samples { margin-top: 20px; padding-top: 16px; border-top: 1px solid #1e293b; text-align: center; }
    .sample-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 10px; }
    .chip { background: #1e293b; border: 1px solid #334155; color: #94a3b8; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.15s; }
    .chip:hover { background: #0284c7; color: white; border-color: #0284c7; }
    
    /* Output Panel */
    .out-box { min-height: 280px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
    .empty-state { color: #64748b; font-size: 14px; }
    .badge-row { display: flex; gap: 10px; align-items: center; justify-content: center; margin-bottom: 12px; }
    .badge { padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px; background: #0284c7; color: white; }
    .badge-style { background: #059669; color: white; }
    .meta-text { color: #64748b; font-size: 12px; margin-bottom: 14px; }
    
    /* Tabs */
    .tab-row { display: flex; gap: 8px; width: 100%; border-bottom: 1px solid #1e293b; margin-bottom: 12px; }
    .tab { padding: 8px 14px; cursor: pointer; font-size: 13px; font-weight: 600; color: #64748b; border-bottom: 2px solid transparent; }
    .tab.active { color: #38bdf8; border-bottom: 2px solid #38bdf8; }
    
    .code-container { width: 100%; text-align: left; position: relative; }
    .code-header { display: flex; justify-content: space-between; align-items: center; background: #0a0e17; padding: 8px 12px; border-top-left-radius: 8px; border-top-right-radius: 8px; border: 1px solid #1e293b; border-bottom: none; font-size: 12px; color: #94a3b8; }
    pre { background: #060911; padding: 14px; margin: 0; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; font-size: 12px; line-height: 1.5; overflow-x: auto; color: #7dd3fc; border: 1px solid #1e293b; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; max-height: 240px; }
    .copy-btn { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 4px 10px; font-size: 11px; border-radius: 4px; cursor: pointer; }
    .copy-btn:hover { background: #334155; color: white; }
    .spinner { display: none; margin: 20px auto; width: 32px; height: 32px; border: 3px solid rgba(56, 189, 248, 0.2); border-top-color: #38bdf8; border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🎨 Sketch2React Microservice</h1>
      <p class="sub">Draw a hand-drawn UI wireframe or click a sample to generate the React component.</p>
    </div>

    <div class="grid">
      <!-- Input Panel -->
      <div class="panel">
        <div class="panel-title">✏️ Input: Drawing Canvas</div>
        <canvas id="canvas" width="227" height="227"></canvas>
        <div class="btn-row">
          <button id="predictBtn">⚡ Convert to React</button>
          <button id="clearBtn" class="secondary">Clear</button>
        </div>
        
        <div class="samples">
          <span style="font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase;">Or pick a test sketch:</span>
          <div class="sample-chips">
            <span class="chip" onclick="loadSample('Button')">Button</span>
            <span class="chip" onclick="loadSample('Card')">Card</span>
            <span class="chip" onclick="loadSample('InputField')">InputField</span>
            <span class="chip" onclick="loadSample('Navbar')">Navbar</span>
            <span class="chip" onclick="loadSample('Modal')">Modal</span>
          </div>
        </div>
      </div>

      <!-- Output Panel -->
      <div class="panel">
        <div class="panel-title">⚡ Output: React Component</div>
        <div id="spinner" class="spinner"></div>
        <div id="outBox" class="out-box">
          <p class="empty-state">Draw on the canvas and click <strong>"Convert to React"</strong>.</p>
        </div>
      </div>
    </div>
  </div>

  <script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let drawing = false;
    let currentData = null;
    let activeTab = 'code';

    function resetCanvas() {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, 227, 227);
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      document.querySelectorAll('.chip').forEach(c => c.style.borderColor = '#334155');
    }
    resetCanvas();

    function getCoords(e) {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      return { x: clientX - rect.left, y: clientY - rect.top };
    }

    function startDraw(e) {
      drawing = true;
      const { x, y } = getCoords(e);
      ctx.beginPath();
      ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(x, y);
    }

    function draw(e) {
      if (!drawing) return;
      if (e.cancelable) e.preventDefault();
      const { x, y } = getCoords(e);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y);
    }

    function stopDraw() {
      if (drawing) {
        drawing = false;
        ctx.beginPath();
      }
    }

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDraw);
    canvas.addEventListener('mouseleave', stopDraw);

    canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDraw(e); }, { passive: false });
    canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); }, { passive: false });
    canvas.addEventListener('touchend', stopDraw);

    document.getElementById('clearBtn').onclick = () => {
      resetCanvas();
      currentData = null;
      document.getElementById('outBox').innerHTML = '<p class="empty-state">Draw on the canvas and click <strong>"Convert to React"</strong>.</p>';
    };

    function loadSample(type) {
      resetCanvas();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 3;
      if (type === 'Button') {
        ctx.strokeRect(40, 90, 145, 45);
        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText('SUBMIT', 80, 118);
      } else if (type === 'Card') {
        ctx.strokeRect(25, 20, 175, 185);
        ctx.strokeRect(40, 35, 145, 70);
        ctx.beginPath(); ctx.moveTo(40, 125); ctx.lineTo(165, 125); ctx.stroke();
        ctx.strokeRect(40, 150, 80, 32);
      } else if (type === 'InputField') {
        ctx.strokeRect(20, 92, 185, 38);
        ctx.fillStyle = '#64748b';
        ctx.font = '12px sans-serif';
        ctx.fillText('Enter email address...', 32, 116);
      } else if (type === 'Navbar') {
        ctx.strokeRect(8, 12, 211, 48);
        ctx.strokeRect(18, 22, 32, 28);
        ctx.beginPath(); ctx.arc(195, 36, 11, 0, Math.PI*2); ctx.stroke();
      } else if (type === 'Modal') {
        ctx.strokeRect(36, 32, 152, 158);
        ctx.strokeRect(50, 144, 52, 28);
        ctx.strokeRect(118, 144, 52, 28);
      }
      runInference();
    }

    document.getElementById('predictBtn').onclick = runInference;

    async function runInference() {
      const spinner = document.getElementById('spinner');
      const outBox = document.getElementById('outBox');
      spinner.style.display = 'block';
      outBox.style.display = 'none';

      const base64 = canvas.toDataURL('image/png');

      try {
        const res = await fetch('/predict-sketch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: base64 })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        currentData = data;
        spinner.style.display = 'none';
        outBox.style.display = 'block';

        renderOutput();
      } catch (err) {
        spinner.style.display = 'none';
        outBox.style.display = 'block';
        outBox.innerHTML = `<p style="color: #f87171;">Inference failed: ${err.message}</p>`;
      }
    }

    function switchTab(tab) {
      activeTab = tab;
      renderOutput();
    }

    function renderOutput() {
      if (!currentData) return;
      const data = currentData;

      if (data.status === 'empty' || data.detected_component === 'Empty Canvas') {
        document.getElementById('outBox').innerHTML = `
          <div style="text-align: center; padding: 24px;">
            <div style="font-size: 32px; margin-bottom: 8px;">✏️</div>
            <h3 style="margin: 0 0 6px 0; color: #facc15; font-size: 16px;">Canvas is Empty</h3>
            <p style="color: #94a3b8; font-size: 13px; max-width: 280px; margin: 0 auto 12px auto; line-height: 1.4;">
              Draw a UI sketch on the canvas or click one of the sample chips below to test!
            </p>
          </div>
        `;
        return;
      }

      const comp = data.detected_component;
      const style = data.style_variant;
      const conf = data.confidence;
      const timeMs = data.processing_time_ms;
      const reactCode = data.generated_react_code || '';
      const prompt = data.suggested_llm_prompt || '';

      const content = activeTab === 'code' ? reactCode : prompt;
      const headerLabel = activeTab === 'code' ? 'React JSX Component' : 'React-Comps LLM Prompt';

      document.getElementById('outBox').innerHTML = `
        <div class="badge-row">
          <span class="badge">${comp}</span>
          <span class="badge badge-style">${style}</span>
        </div>
        <div class="meta-text">Confidence: <strong>${conf}</strong> • Latency: <strong>${timeMs}</strong></div>
        
        <div class="tab-row">
          <div class="tab ${activeTab === 'code' ? 'active' : ''}" onclick="switchTab('code')">⚛️ React JSX Code</div>
          <div class="tab ${activeTab === 'prompt' ? 'active' : ''}" onclick="switchTab('prompt')">💬 LLM Prompt</div>
        </div>

        <div class="code-container">
          <div class="code-header">
            <span>${headerLabel}</span>
            <button id="copyBtn" class="copy-btn" onclick="copyCurrent()">Copy ${activeTab === 'code' ? 'Code' : 'Prompt'}</button>
          </div>
          <pre id="displayedText">${escapeHtml(content)}</pre>
        </div>
      `;
    }

    function escapeHtml(text) {
      return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function copyCurrent() {
      const text = document.getElementById('displayedText').innerText;
      navigator.clipboard.writeText(text);
      const btn = document.getElementById('copyBtn');
      if (btn) {
        const orig = btn.innerText;
        btn.innerText = '✓ Copied!';
        btn.style.background = '#059669';
        btn.style.color = 'white';
        setTimeout(() => {
          btn.innerText = orig;
          btn.style.background = '';
          btn.style.color = '';
        }, 1500);
      }
    }
  </script>
</body>
</html>
"""


@app.post("/predict-sketch")
async def predict_sketch(request: Request):
  """Analyzes a hand-drawn sketch (Base64 JSON or uploaded file) and returns predicted React component."""
  pipeline = get_pipeline()
  img_np = None
  content_type = request.headers.get("content-type", "")

  # 1. Handle multipart/form-data file uploads
  if "multipart/form-data" in content_type:
    form = await request.form()
    file = form.get("file")
    if file and hasattr(file, "read"):
      contents = await file.read()
      nparr = np.frombuffer(contents, np.uint8)
      img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
      if img_np is not None:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

  # 2. Handle JSON payload (Base64 canvas image)
  if img_np is None:
    try:
      payload = await request.json()
    except Exception:
      payload = {}

    if isinstance(payload, dict):
      b64_str = payload.get("image_base64", "")
      if b64_str:
        encoded = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
        try:
          contents = base64.b64decode(encoded)
          nparr = np.frombuffer(contents, np.uint8)
          img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
          if img_np is not None:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        except Exception:
          pass

  if img_np is None:
    raise HTTPException(
        status_code=400,
        detail="No valid image received. Send a multipart 'file' upload or JSON {\"image_base64\": \"data:image/png;base64,...\"}.",
    )

  try:
    result = pipeline.process(img_np)
    return result
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))



@app.get("/labs")
def get_labs_revision_guide():
  """Returns full revision syllabus notes and mathematical formulas for DL Labs 0 to 8."""
  return {
      "Lab 0": {
          "topic": "Linear Algebra & NumPy Foundations",
          "equations": ["Orthogonality: A . A^T = I", "Determinant via Gaussian Elimination Upper Triangular: det = prod(U_ii)", "Trace: sum(A_ii)"],
          "usage_in_service": "Straightening hand-drawn strokes with 2D orthogonal rotation matrices and verifying zero shape distortion."
      },
      "Lab 1": {
          "topic": "TensorFlow Computation Graphs",
          "equations": ["tf.constant (immutable tensors)", "Sequential([Input, Dense, Dense])"],
          "usage_in_service": "Core tensor conversion and graph execution pipeline."
      },
      "Lab 2": {
          "topic": "Backpropagation from Scratch",
          "equations": ["Sigmoid: 1 / (1 + exp(-x))", "Derivative: out * (1 - out)", "Delta: delta_L = (y - y_hat) * sig'(y_hat)"],
          "usage_in_service": "Confidence score calibration and mathematical gradient validation without libraries."
      },
      "Lab 3": {
          "topic": "Batch Normalization",
          "equations": ["x_hat = (x - mu_B) / sqrt(var_B + eps)", "y = gamma * x_hat + beta"],
          "usage_in_service": "Normalizing pixel contrast and feature activations across different pen thickness and paper backgrounds."
      },
      "Lab 4": {
          "topic": "Perceptron & -w0 Threshold Derivation",
          "equations": ["Decision rule: w1*x1 + w2*x2 >= -w0", "Lower bound: max_{y=0}(w.x) < -w0 <= min_{y=1}(w.x) : Upper bound"],
          "usage_in_service": "Rapid geometric layout gating (e.g. Button vs Card based on aspect ratio and bounding box area)."
      },
      "Lab 5": {
          "topic": "Deep ANN with Dropout (0.2) on MNIST",
          "equations": ["Dropout(p=0.2) randomly zeroes activations during training and scales by 1/(1-p)"],
          "usage_in_service": "Predicting component styling variants (Primary, Outline, Ghost, Danger) while preventing overfitting on messy strokes."
      },
      "Lab 6": {
          "topic": "CNN & Custom tf.GradientTape Mini-Batch Loop",
          "equations": ["with tf.GradientTape() as tape:", "grads = tape.gradient(loss, vars)", "optimizer.apply_gradients(zip(grads, vars))"],
          "usage_in_service": "Training stroke edge and contour detection layers with class-weighted loss."
      },
      "Lab 7": {
          "topic": "AlexNet Architecture from Scratch & Custom LRN",
          "equations": ["LRN: b = a / (bias + alpha * sum(a_j^2))^beta", "Overlapping MaxPool: 3x3 with stride 2", "11x11 Conv (stride 4)"],
          "usage_in_service": "Primary visual backbone classifying wireframe images into 5 React UI component categories."
      },
      "Lab 8": {
          "topic": "Unsupervised Pre-training with Stacked Autoencoders",
          "equations": ["Vanishing gradient: random init gets stuck at 11% accuracy", "Layer-wise MSE reconstruction: x -> h -> x_hat gives >96% accuracy"],
          "usage_in_service": "Pre-training visual representations on thousands of unlabeled sketches to overcome small dataset limitations."
      }
  }


@app.get("/fraz-algos")
def get_fraz_algorithms_summary():
  """Returns summary of Fraz DSA Sheet algorithms used in the service."""
  return {
      "2D Prefix Sum (LC 304)": "O(1) bounding box stroke density queries for rapid contour validation",
      "Two Pointers": "Scanning 1D projection profiles from left and right to find tight component bounds in O(N)",
      "Monotonic Stack (LC 84)": "Largest Rectangle in Histogram to detect maximal parent layout containers/cards",
      "Merge Intervals (LC 56)": "Non-Maximum Suppression (NMS) merging overlapping candidate bounding boxes in O(N log N)",
      "Graph BFS (LC 200)": "Connected components / Number of Islands grouping disjoint drawn strokes into single elements",
      "Kahn's Topological Sort (LC 210)": "Ordering nested React UI components in valid parent-to-child JSX rendering order",
      "Binary Search": "Lower bound binary search for dynamic confidence thresholding in O(log N)",
      "Trie (Prefix Tree)": "Instant prefix auto-completion for React component names and Tailwind CSS classes in O(L)",
      "Top-K Heaps (LC 347)": "Min-heap of size K to rank the top-k most confident component predictions in O(N log K)",
      "Dynamic Programming (LC 72)": "Edit distance aligning predicted component sequences to standard design system templates in O(M*N)"
  }


if __name__ == "__main__":
  import uvicorn
  print("\nStarting Sketch2React FastAPI Microservice on http://localhost:8000 ...")
  uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
