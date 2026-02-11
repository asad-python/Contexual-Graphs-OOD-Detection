# Contextual Graphs for Out-of-Distribution Detection in nuScenes

This repository implements a complete, end-to-end framework for evaluating Out-of-Distribution (OOD) detection in autonomous driving scenes using the **nuScenes** dataset.

The project provides a full research pipeline including:

1. Creation of an **OOD-augmented nuScenes dataset**
2. Evaluation of standard object detectors for OOD robustness  
3. A **contextual graph-based OOD detection method**
4. Visualization of quantitative and qualitative results  


---

## Project Goal

Modern object detectors are trained on a fixed set of classes. In real-world autonomous driving, vehicles often encounter **unknown or novel objects** that were not part of the training data.

The goal of this research is to evaluate:

- How pretrained perception systems react to novel objects  
- Whether contextual reasoning can help detect anomalous objects  
- How reliable common detectors are under OOD conditions  

To achieve this, we:

- Insert synthetic novel objects into nuScenes camera images  
- Measure how detectors react to them  
- Apply graph-based reasoning to detect contextual anomalies  

---

## Repository Structure

After cloning the repository, the project structure should look like this:

```text
contextual-graphs-ood/
│
├── config/
│   └── defaults.py            # Central configuration file
│
├── ood_addition.py            # Creates OOD-augmented dataset
├── detection.py               # Runs detectors and computes metrics
├── context_graph.py           # Graph-based OOD detection
├── results_viz.py             # Visualization and analysis
│
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation
```

---

## Installation

### 1) Clone the Repository

```bash
git clone <URL>
cd contextual-graphs-ood
```

### 2) Create a Virtual Environment (We created it in Python 3+)

```bash
python -m venv venv
source venv/bin/activate       # Linux / Mac
# venv\Scripts\activate        # Windows
```

### 3) Install Dependencies

```bash
pip install -r requirements.txt
```

### Important Note on PyTorch Geometric

The contextual graph pipeline relies on **torch-geometric**. Installation of PyG depends on your specific PyTorch and CUDA versions.

If you face installation problems, follow the official guide:

https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Dataset Preparation

### nuScenes Dataset

You must have the nuScenes dataset available locally, You can download it from (https://www.nuscenes.org/nuscenes#download).


### OOD Patch Assets

Create a directory containing transparent PNG images of novel objects from MS-COCO dataset (https://cocodataset.org/#home):

```text
ASSETS_DIR/
   object1.png
   object2.png
   object3.png
   ...
```

Requirements:

- Transparent background (RGBA PNG)
- Realistic cropped object images
- Preferably diverse object types, Related to Autonomous vehicles.

---

## Configuration

All paths are defined in:

```text
config/defaults.py
```

You MUST update the placeholders with your local paths:

```python
SRC_DATASET = "WRITE_SOURCE_NUSCENES_DATASET_PATH_HERE"
DST_DATASET = "WRITE_DESTINATION_DATASET_PATH_HERE"
ASSETS_DIR  = "WRITE_ASSETS_DIRECTORY_PATH_HERE"
DEBUG_OUTPUT_DIR = "WRITE_DEBUG_OUTPUT_PATH_HERE"
```

### Meaning of Each Path

| Variable | Description |
|--------|-------------|
| SRC_DATASET | Original nuScenes dataset path |
| DST_DATASET | Where modified dataset will be created |
| ASSETS_DIR | Folder containing OOD patch PNGs |
| DEBUG_OUTPUT_DIR | Optional folder for debugging outputs |

---

## OOD Patch Default

By default, pasted objects follow augmentation rules designed to keep them realistic:

- Objects are pasted only in the **lower half** of images  
- Random scale between **0.3 – 0.6**
- Rotation within **±8 degrees**
- Brightness and contrast 
- Horizontal flip  

These defaults can be modified inside:

- `config/defaults.py`
- `ood_addition.py`

---

# Running the Full Pipeline

The pipeline must be executed in the following order.

---

## STEP 1 – Create OOD-Augmented Dataset

```bash
python ood_addition.py
```

### What This Script Does

- Copies the dataset from `SRC_DATASET` to `DST_DATASET`
- Pastes novel objects into images
- Generates two ground-truth JSON files

### Outputs

```
DST_DATASET/v1.0-mini/detection_id.json
DST_DATASET/v1.0-mini/detection_novel.json
```

### Output JSON Formats

#### detection_id.json

Contains projected 2D bounding boxes for original nuScenes objects:

```json
{
  "results": {
    "sample_data_token": [
      {
        "bbox_2d": [x1, y1, x2, y2],
        "detection_name": "car",
        "detection_score": 1.0,
        "token": "unique-id"
      }
    ]
  }
}
```

#### detection_novel.json

Contains bounding boxes for pasted OOD objects:

```json
{
  "results": {
    "sample_data_token": [
      {
        "bbox_2d": [x1, y1, x2, y2],
        "detection_name": "novel",
        "detection_score": 1.0,
        "token": "unique-id"
      }
    ]
  }
}
```

These two files are used as ground truth for all following steps.

---

## STEP 2 – Evaluate Object Detectors

```bash
python detection.py
```

### Purpose

This script evaluates how well pretrained detectors handle:

- Normal in-distribution objects  
- Newly pasted novel OOD objects  

### Evaluated Object Detectors

The following models are evaluated in this project:

#### Torchvision Detectors

- Faster R-CNN (ResNet50)
- Faster R-CNN (MobileNetV3)
- RetinaNet (ResNet50)
- SSDLite (MobileNetV3)
- SSD300 (VGG16)

#### YOLO Detectors

- YOLOv5s
- YOLOv5m
- YOLOv8n
- YOLOv8s
- YOLOv8m
- YOLOv8l
- YOLOv11n
- YOLOv11s

#### Transformer-based Detectors

- DETR (ResNet50)
- DETR (ResNet50-DC5)
- Deformable DETR (ResNet50)

#### Proposed Method

- Contextual Graph OOD Detection (Ours)

### Metrics Computed

| Metric | Description |
|------|-------------|
| AP@0.5 | Detection quality on ID objects |
| Precision / Recall | Standard detection metrics |
| OOD_FP | Fraction of detections overlapping OOD GT |
| AUROC | Ability to separate ID vs OOD |
| FPR@95 | False positive rate at 95% recall |

### Output

Creates:

```text
results/ood_report.csv
```

This CSV contains:

- Overall metrics per model  
- Per-camera metrics  
- Camera-averaged performance  

---

## STEP 3 – Contextual Graph OOD Detection

```bash
python context_graph.py
```

### What This Script Does

- Builds a graph for every camera frame  
- Extracts visual embeddings using DINO (via timm)  
- Adds geometric features  
- Connects nodes using KNN  
- Trains a Variational Graph Autoencoder (VGAE) using ID data  
- Scores nodes using:
  - VGAE reconstruction error  
  - Contextual Mahalanobis distance  

### Output

```
out/context_graph_scores.json
```

Example entry:

```json
{
  "sd_token": "sample_data_token",
  "channel": "CAM_FRONT",
  "bbox_2d": [x1, y1, x2, y2],
  "label": 1,
  "vgae_score": 2.14
}
```

### Embedding Caching

Crop embeddings are cached to speed up repeated runs:

```
<DST_DATASET>/.cache/emb_v1/
```

---

## STEP 4 – Visualize Results

```bash
python results_viz.py
```

## Qualitative Examples

### OOD Injection and Detection

![Detection Visualization](dete (1).png)

### Context Graph Visualization

![Graph Visualization](graphs_sixview (2))

### Sample OOD Augmented Images

![OOD Examples](ood_grid (2))


