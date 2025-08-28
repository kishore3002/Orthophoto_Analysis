# app/app.py
import os
import traceback
from typing import Optional, Tuple
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import gradio as gr

# Try to import YOLO (ultralytics). If you don't want it now, the app will show a message.
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---- Paths ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_DIR = os.path.join(BASE_DIR, "test_samples", "input_images")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# ---- Target canonical colors (RGB) ----
TARGET_COLORS = {
    "building_rcc": (255, 255, 255),   # white
    "building_tiled": (255, 0, 0),     # red
    "building_tin": (0, 0, 255),       # blue
    "street_road": (150, 75, 0),       # brown
    "main_road": (128, 128, 128),      # grey
    "waterbody": (0, 255, 255)         # sky blue
}
LEGEND_ORDER = ["building_rcc", "building_tiled", "building_tin",
                "main_road", "street_road", "waterbody"]

# ---- Helper: canonicalize model class name to our target name ----
def _canon(name: str) -> str:
    s = name.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    if "main" in s and "road" in s:
        return "main_road"
    if "street" in s or (("road" in s) and ("main" not in s)):
        return "street_road"
    if "tiled" in s or "tile" in s:
        return "building_tiled"
    if "tin" in s:
        return "building_tin"
    if "rcc" in s or ("concrete" in s and "build" in s):
        return "building_rcc"
    if "water" in s or "lake" in s or "pond" in s:
        return "waterbody"
    return s

# ---- Load sample images automatically ----
def get_sample_map(sample_dir: str) -> dict:
    if not os.path.isdir(sample_dir):
        return {}
    files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = sorted(files)
    return {os.path.basename(f): os.path.join(sample_dir, f) for f in files}

SAMPLE_MAP = get_sample_map(SAMPLE_DIR)  # e.g., {'tile_24.jpg': '.../tile_24.jpg', ...}

# ---- Load YOLO model (if possible) ----
model = None
names = {}
ID2CANON = {}
ID2COLOR = {}

if YOLO is None:
    print("ultralytics.YOLO not installed or failed to import. Install 'ultralytics' to enable segmentation.")
else:
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        names = model.names  # dict: id -> class_name
        # Build ID->canonical name and ID->color maps
        ID2CANON = {i: _canon(n) for i, n in names.items()}
        ID2COLOR = {i: TARGET_COLORS.get(ID2CANON[i], (0, 0, 0)) for i in names.keys()}
        print("Model loaded. Classes:", names)
    except Exception as e:
        print("Failed to load YOLO model:", e)
        traceback.print_exc()
        model = None

# ---- Utility: create an image (PIL) with an error message (returned to UI) ----
def make_error_image(msg: str, size: Tuple[int, int] = (900, 450)) -> np.ndarray:
    img = Image.new("RGB", size, color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    lines = msg.splitlines()
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=(255, 80, 80), font=font)
        y += 22
    return np.array(img)

# ---- Core segmentation: input is RGB numpy array, returns RGB numpy array (visualization) ----
def segment_np(img_rgb: np.ndarray) -> np.ndarray:
    try:
        if model is None:
            return make_error_image("Model not loaded. Place your weights at:\n" + MODEL_PATH)

        # Ensure uint8 RGB
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)

        counts = {k: 0 for k in LEGEND_ORDER}
        results = model(img_rgb)  # ultralytics accepts numpy RGB

        overlay = np.zeros_like(img_rgb, dtype=np.uint8)

        res0 = results[0]

        # Collect masks and class ids safely
        masks_arr = None
        class_ids = []
        if getattr(res0, "masks", None) is not None and getattr(res0.masks, "data", None) is not None:
            masks_arr = res0.masks.data.cpu().numpy()  # shape [N, Hm, Wm]
        if getattr(res0, "boxes", None) is not None and getattr(res0.boxes, "cls", None) is not None:
            class_ids = res0.boxes.cls.cpu().numpy().astype(int).tolist()

        if masks_arr is not None and len(masks_arr) > 0:
            for i, mask in enumerate(masks_arr):
                cid = class_ids[i] if i < len(class_ids) else None
                cname = names.get(cid, str(cid)) if cid is not None else str(i)
                canon = ID2CANON.get(cid, _canon(cname))
                color = ID2COLOR.get(cid, (0, 0, 0))

                if canon in counts:
                    counts[canon] += 1

                # resize mask to image size
                m_resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                m_bin = (m_resized > 0.5)
                overlay[m_bin] = color

        # Single blend to avoid color mixing
        blended = cv2.addWeighted(img_rgb, 1.0, overlay, 0.5, 0)

        # Build figure: left image, right legend
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(blended)
        ax[0].axis("off")
        ax[0].set_title("Segmentation Result")

        legend_elements = []
        for key in LEGEND_ORDER:
            color = TARGET_COLORS[key]
            rgb_norm = tuple(c / 255 for c in color)
            cnt = counts.get(key, 0)
            brightness = sum(color) / 3
            patch_kwargs = dict(facecolor=rgb_norm, label=f"{key}: {cnt}")
            if brightness > 220:
                patch_kwargs.update(edgecolor="black", linewidth=1.5)
            legend_elements.append(Patch(**patch_kwargs))

        ax[1].legend(handles=legend_elements, loc="center", frameon=False, fontsize=12)
        ax[1].axis("off")
        ax[1].set_title("Detected Object Counts")
        plt.tight_layout()

        # fig.canvas.draw()
        # out = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # out = out.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # plt.close(fig)
        # return out
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        out = np.asarray(buf)
        plt.close(fig)
        return out

    except Exception as e:
        tb = traceback.format_exc()
        print("Segmentation error:", e, tb)
        return make_error_image("Segmentation error:\n" + str(e) + "\n\nSee server console for details.")

# ---- Inference wrapper: accepts dropdown choice (filename) and uploaded file path ----
def inference(sample_choice: Optional[str], uploaded_path: Optional[str]):
    try:
        # priority: uploaded image
        if uploaded_path:
            # uploaded_path is a filesystem path
            img = cv2.imread(uploaded_path)
            if img is None:
                return make_error_image("Failed to read uploaded image.")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return segment_np(img_rgb)

        # else use sample
        if sample_choice and sample_choice in SAMPLE_MAP:
            sample_path = SAMPLE_MAP[sample_choice]
            img = cv2.imread(sample_path)
            if img is None:
                return make_error_image(f"Failed to read sample image:\n{sample_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return segment_np(img_rgb)

        return make_error_image("No image provided. Upload or pick a sample image.")
    except Exception as e:
        tb = traceback.format_exc()
        print("Inference wrapper error:", tb)
        return make_error_image("Internal error:\n" + str(e))

# Function to get model status info
def get_model_status():
    if model is None:
        return "❌ Model Not Loaded", "red"
    else:
        return "✅ Model Ready", "green"

# Function to get sample count info
def get_sample_info():
    count = len(SAMPLE_MAP)
    return f"{count} Sample Images Available", "blue" if count > 0 else "orange"

# Enhanced CSS for professional look with BETTER TEXT VISIBILITY
custom_css = """
/* Global Styles */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* Header Styles */
.hero-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    color: white !important;
}

.hero-subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
    margin-bottom: 0;
    color: white !important;
}

/* Card Styles - FIXED FOR BETTER VISIBILITY */
.info-card {
    background: white !important;
    color: #2c3e50 !important;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
    transition: transform 0.2s ease;
}

.info-card:hover {
    transform: translateY(-2px);
}

.info-card * {
    color: #2c3e50 !important;
}

.feature-card {
    background: white !important;
    color: #2c3e50 !important;
    border-radius: 10px;
    padding: 1.25rem;
    margin: 0.5rem;
    border: 1px solid #e9ecef;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    border-color: #667eea;
}

.feature-card * {
    color: #2c3e50 !important;
}

.feature-card h3 {
    color: #667eea !important;
    margin-top: 0 !important;
    font-weight: 600 !important;
}

.status-card {
    background: white !important;
    color: #2c3e50 !important;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #28a745;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.status-card * {
    color: #2c3e50 !important;
}

.status-card h3 {
    font-weight: 600 !important;
    margin-top: 0 !important;
}

.legend-card {
    background: white !important;
    color: #2c3e50 !important;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #6c757d;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.legend-card * {
    color: #2c3e50 !important;
}

.legend-card h3 {
    font-weight: 600 !important;
    margin-top: 0 !important;
}

/* Section Headers */
.section-header {
    font-size: 1.8rem;
    font-weight: 600;
    color: #2c3e50 !important;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #667eea;
}

.subsection-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #34495e !important;
    margin: 1.5rem 0 0.5rem 0;
}

/* Button Styles */
.custom-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    color: white !important;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.custom-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

/* Analysis Section */
.analysis-section {
    background: #ffffff !important;
    color: #2c3e50 !important;
    border-radius: 15px;
    padding: 2rem;
    margin: 2rem 0;
    box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    border: 1px solid #e9ecef;
}

.input-section {
    background: white !important;
    color: #2c3e50 !important;
    border-radius: 10px;
    padding: 1.5rem;
    border: 1px solid #dee2e6;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.input-section * {
    color: #2c3e50 !important;
}

.results-section {
    background: white !important;
    color: #2c3e50 !important;
    border-radius: 10px;
    padding: 1.5rem;
    border: 1px solid #dee2e6;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.results-section * {
    color: #2c3e50 !important;
}

/* Step Guide Styles */
.step-guide {
    background: white !important;
    color: #2c3e50 !important;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.step-guide * {
    color: #2c3e50 !important;
}

.step-guide h4 {
    color: #667eea !important;
    font-weight: 600 !important;
    margin-top: 0 !important;
}

.step-guide ol, .step-guide ul {
    line-height: 1.8 !important;
    padding-left: 1.5rem !important;
}

.step-guide li {
    margin: 0.5rem 0 !important;
    color: #2c3e50 !important;
}

/* Result info styles */
.result-info {
    background: #e8f4f8 !important;
    color: #0c5460 !important;
    padding: 1rem;
    border-radius: 8px;
    margin-top: 1rem;
    border: 1px solid #b8e6f1;
}

.result-info * {
    color: #0c5460 !important;
}

.result-info h4 {
    margin-top: 0 !important;
    font-weight: 600 !important;
}

/* Responsive Design */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
    }
    
    .feature-card {
        margin: 0.25rem;
    }
}

/* Footer */
.footer-section {
    background: #2c3e50 !important;
    color: white !important;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    margin-top: 3rem;
}

.footer-section * {
    color: white !important;
}

/* Force text visibility in all elements */
p, div, span, h1, h2, h3, h4, h5, h6, li, td, th {
    color: inherit !important;
}

/* Dark mode overrides */
.dark .info-card,
.dark .feature-card,
.dark .status-card,
.dark .legend-card,
.dark .input-section,
.dark .results-section,
.dark .step-guide,
.dark .result-info {
    background: white !important;
    color: #2c3e50 !important;
}

.dark .info-card *,
.dark .feature-card *,
.dark .status-card *,
.dark .legend-card *,
.dark .input-section *,
.dark .results-section *,
.dark .step-guide *,
.dark .result-info * {
    color: #2c3e50 !important;
}
"""

# ---- Build Enhanced Gradio UI with Sections ----
sample_choices = list(SAMPLE_MAP.keys())
default_choice = sample_choices[0] if sample_choices else None

with gr.Blocks(title="AI-Powered Drone Imagery Segmentation", css=custom_css) as demo:
    
    # ==================== HERO SECTION ====================
    gr.HTML("""
    <div class="hero-section">
        <h1 class="hero-title">🛰️ AI-Powered Drone Imagery Segmentation</h1>
        <p class="hero-subtitle">Advanced computer vision system for automatic analysis of aerial drone imagery</p>
    </div>
    """)
    
    # ==================== ABOUT SECTION ====================
    with gr.Row():
        gr.HTML("""
        <div class="info-card">
            <h2 class="section-header">🎯 About This Application</h2>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                This cutting-edge AI tool leverages state-of-the-art computer vision to automatically analyze drone imagery 
                and identify urban infrastructure elements. Perfect for urban planners, civil engineers, and geographical analysts.
            </p>
        </div>
        """)
    
    # ==================== FEATURES SECTION ====================
    gr.HTML('<h2 class="section-header">🚀 Key Capabilities</h2>')
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="feature-card">
                <h3>🏢 Smart Building Detection</h3>
                <p>Automatically identifies and classifies different building types including RCC, tiled, and tin roof structures with precise color coding.</p>
            </div>
            """)
        
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="feature-card">
                <h3>🛣️ Advanced Road Classification</h3>
                <p>Intelligently distinguishes between main highways and local street roads using sophisticated pattern recognition.</p>
            </div>
            """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="feature-card">
                <h3>💧 Water Body Detection</h3>
                <p>Precisely maps lakes, ponds, rivers, and other water bodies within the aerial imagery for comprehensive analysis.</p>
            </div>
            """)
        
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="feature-card">
                <h3>📊 Real-time Analytics</h3>
                <p>Provides instant statistical analysis with accurate counts of each detected infrastructure element.</p>
            </div>
            """)
    
    # ==================== SYSTEM STATUS SECTION ====================
    gr.HTML('<h2 class="section-header">📊 System Status</h2>')
    
    model_status_text, model_color = get_model_status()
    sample_info_text, sample_color = get_sample_info()
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(f"""
            <div class="status-card">
                <h3>🤖 AI Model Status</h3>
                <p><strong>{model_status_text}</strong></p>
                <p style="margin: 0;">Loaded with {len(names)} detection classes</p>
            </div>
            """)
        
        with gr.Column(scale=1):
            gr.HTML(f"""
            <div class="status-card">
                <h3>📁 Sample Dataset</h3>
                <p><strong>{sample_info_text}</strong></p>
                <p style="margin: 0;">Ready for demonstration and testing</p>
            </div>
            """)
        
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="legend-card">
                <h3>🎨 Detection Legend</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; font-size: 0.9rem;">
                    <div><span style="color: blue;">🔵</span> Tin Buildings</div>
                    <div><span style="color: red;">🔴</span> Tiled Buildings</div>
                    <div><span style="color: black;">⚪</span> RCC Buildings</div>
                    <div><span style="color: brown;">🟫</span> Street Roads</div>
                    <div><span style="color: gray;">⚫</span> Main Roads</div>
                    <div><span style="color: cyan;">🔵</span> Water Bodies</div>
                </div>
            </div>
            """)
    
    # ==================== HOW TO USE SECTION ====================
    gr.HTML('<h2 class="section-header">📋 How to Use</h2>')
    
    with gr.Row():
        gr.HTML("""
        <div class="step-guide">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
                <div>
                    <h4>📝 Step-by-Step Guide</h4>
                    <ol>
                        <li><strong>Choose Input:</strong> Select sample or upload image</li>
                        <li><strong>Run Analysis:</strong> Click segmentation button</li>
                        <li><strong>View Results:</strong> Check color-coded output</li>
                        <li><strong>Interpret Data:</strong> Use legend for understanding</li>
                    </ol>
                </div>
                <div>
                    <h4>💡 Pro Tips</h4>
                    <ul>
                        <li>Use high-resolution images (1080p+)</li>
                        <li>Ensure good lighting and contrast</li>
                        <li>Optimal altitude: 50-200 meters</li>
                        <li>Supported formats: JPG, PNG</li>
                    </ul>
                </div>
            </div>
        </div>
        """)
    
    # ==================== MAIN ANALYSIS INTERFACE ====================
    gr.HTML('<h2 class="section-header">🔬 Analysis Interface</h2>')
    
    with gr.Row():
        # LEFT PANEL - INPUT CONTROLS
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="input-section">
                <h3 style="margin-top: 0;">📤 Input Selection</h3>
            </div>
            """)
            
            with gr.Group():
                gr.Markdown("### 🎯 Choose Sample Image")
                sample_dropdown = gr.Dropdown(
                    choices=sample_choices,
                    value=default_choice,
                    label="Pre-loaded Test Images",
                    interactive=True
                )
                
                gr.Markdown("### 📁 Or Upload Your Image")
                uploaded = gr.Image(
                    type="filepath", 
                    label="Upload Drone Imagery",
                    interactive=True
                )
                
                with gr.Row():
                    run_btn = gr.Button("🚀 Run Analysis", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)
                
                processing_status = gr.Markdown("**Status:** Ready to process images...")
        
        # RIGHT PANEL - RESULTS
        with gr.Column(scale=2):
            gr.HTML("""
            <div class="results-section">
                <h3 style="margin-top: 0;">📊 Analysis Results</h3>
            </div>
            """)
            
            output = gr.Image(
                label="Segmentation Output",
                interactive=False
            )
            
            with gr.Row():
                gr.HTML("""
                <div class="result-info">
                    <h4>📖 Understanding Results</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <strong>Left Panel:</strong> Original image with AI-detected segments overlaid in colors
                        </div>
                        <div>
                            <strong>Right Panel:</strong> Statistical breakdown showing count of each detected element
                        </div>
                    </div>
                </div>
                """)
    
    # Event handlers with enhanced feedback
    def run_analysis_with_feedback(sample_choice, uploaded_path):
        processing_status = "🔄 **Status:** Analyzing image with AI model..."
        result = inference(sample_choice, uploaded_path)
        success_status = "✅ **Status:** Analysis complete! Results displayed above."
        return result, success_status
    
    def clear_all():
        return None, None, "🧹 **Status:** Interface cleared. Ready for new analysis."
    
    # Event bindings
    run_btn.click(
        fn=run_analysis_with_feedback,
        inputs=[sample_dropdown, uploaded], 
        outputs=[output, processing_status]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[uploaded, output, processing_status]
    )
    
    sample_dropdown.change(
        fn=lambda x: (inference(x, None), "🔄 **Status:** Sample image processed automatically"),
        inputs=sample_dropdown, 
        outputs=[output, processing_status]
    )
    
    # Load default on startup
    if default_choice:
        demo.load(
            fn=lambda: (inference(default_choice, None), "✨ **Status:** Default sample loaded successfully"),
            outputs=[output, processing_status]
        )
    
    # ==================== FOOTER SECTION ====================
    gr.HTML("""
    <div class="footer-section">
        <h3 style="margin-top: 0;">🤖 Powered by Advanced AI Technology</h3>
        <p>Built with YOLO Computer Vision • Gradio Interface • Python Backend</p>
        <p style="margin-bottom: 0; opacity: 0.8;">For urban planning, infrastructure monitoring, and geographical analysis</p>
    </div>
    """)

# ---- Launch ----
if __name__ == "__main__":
    # Quick sanity prints
    print("="*60)
    print("🌐 Starting web interface...")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )
    print(f"🚀 DRONE IMAGERY SEGMENTATION APPLICATION")
    print("="*60)
    print(f"📁 Sample Images: {len(SAMPLE_MAP)} found")
    if SAMPLE_MAP:
        print(f"   Files: {', '.join(list(SAMPLE_MAP.keys())[:3])}{'...' if len(SAMPLE_MAP) > 3 else ''}")
    print(f"🤖 AI Model: {'✅ Loaded Successfully' if model is not None else '❌ Failed to Load'}")
    if model:
        print(f"   Classes: {len(names)} detection categories")
    print(f"📍 Model Path: {MODEL_PATH}")
    print(f"📂 Samples Directory: {SAMPLE_DIR}")
    print("="*60)
    print("🌐 Starting web interface...")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )