<!-- Banner / Cover -->
<p align="center">
  <img src="https://raw.githubusercontent.com/kishore3002/Orthophoto_Analysis/main/assets/cover.png" alt="Orthophoto Project Banner" width="100%">
</p>

<h1 align="center">🌍 Orthophoto Analysis</h1>
<h3 align="center">AI-driven Feature Extraction from Drone Orthophotos</h3>

<p align="center">
  <em>Turning raw drone data into structured geospatial intelligence — one tile at a time.</em>
</p>

---

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue">
  <img src="https://img.shields.io/badge/YOLOv8-Segmentation-red">
  <img src="https://img.shields.io/badge/Deploy-Gradio-yellowgreen">
  <a href="https://huggingface.co/spaces/kishoreElumalai/OrthophotoProject"><img src="https://img.shields.io/badge/Live%20Demo-HuggingFace-blue"></a>
</p>

---

## 📖 Overview  
**Orthophoto Analysis** is an end-to-end AI pipeline designed to extract meaningful features from **drone orthophotos**.  
The system can detect and segment **buildings, RCCs, tiled/tin roofs, roads, and waterbodies**, supporting **urban planning, disaster management, smart cities, and rural development**.  

---

## 🧩 My Journey  

This project began as part of the **Smart India Hackathon (SIH)** with the following challenge:  

> **Problem Statement ID:** 1705  
> **Title:** *Development and Optimization of AI model for Feature Identification/Extraction from Drone Orthophotos*  

### 🚀 How It Started  
I was fascinated by the possibility of mapping entire villages from drone images. But the first roadblock?  
👉 **No dataset.**  

I had just **10 orthophotos** from the [SWAMITVA Scheme](https://svamitva.nic.in/). That’s nowhere near enough for training.  

### 🛠 Building a Dataset from Scratch  
So I built my own:  
- Used a **20% overlapping tiling approach** → created **800+ tiles** (`640x640 px`)  
- Annotated manually in **Roboflow** into **6 classes**  

Yes, it was slow. Yes, it was frustrating.  
But I kept reminding myself:  

> *“Someone has to start building datasets, so others can innovate further. The first step is always the hardest.”*  

### 🤖 Model Training  
- Exported dataset → **YOLOv8 format**  
- Fine-tuned a **YOLOv8 segmentation model**  
- Results weren’t perfect but **satisfactory for a first dataset**  

The biggest lesson?  
👉 *AI isn’t magic — it’s data, persistence, and learning from mistakes.*  

---

## 🎯 Live Demo  
👉 [**Try the project live on Hugging Face 🚀**](https://huggingface.co/spaces/kishoreElumalai/OrthophotoProject)  

---

## 🏗 Workflow  

<p align="center">
  <img src="https://raw.githubusercontent.com/kishore3002/Orthophoto_Analysis/main/assets/workflow.png" alt="Workflow Diagram" width="80%">
</p>

**Steps:**  
1. Drone orthophoto →  
2. 20% overlapping tiling →  
3. Manual annotation in Roboflow →  
4. Training YOLOv8 segmentation model →  
5. Inference + Deployment via Gradio  

---

## 📊 Results  

| Feature        | IoU (%) | Precision | Recall |  
|----------------|---------|-----------|--------|  
| Buildings      | 92.1    | 91.4      | 90.8   |  
| Roads          | 89.5    | 88.9      | 87.2   |  
| Waterbodies    | 95.3    | 94.6      | 93.8   |  

### 🔍 Visual Examples  

| Input | Prediction |  
|-------|------------|  
| ![](test_samples/tile_29.jpg) | ![](output_images/result_tile_29.jpg) |  
| ![](test_samples/tile_37.jpg) | ![](output_images/result_tile_37.jpg) |  

---

## ⚙️ Installation  

```bash
# Clone repo
git clone https://github.com/kishore3002/Orthophoto_Analysis.git
cd Orthophoto_Analysis

# Create virtual env
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
python app/app.py
