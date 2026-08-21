# 🦴 AI Knee Osteoarthritis Detection

An end-to-end Deep Learning application for automated **Knee Osteoarthritis (OA) severity grading** from knee X-ray images.

The system uses an **Ordinal ResNet50** architecture to predict osteoarthritis severity from **Grade 0 to Grade 4**, provides class probabilities and confidence scores, and generates **Grad-CAM visual explanations** to highlight image regions influencing the prediction.

The project is designed as a production-oriented ML system rather than only a model-training experiment, including API serving, interactive deployment, model version verification, monitoring, automated testing, and containerization.

---

## 🚀 Key Features

- 🧠 ResNet50-based deep learning model
- 📊 Ordinal classification for OA severity
- 🦴 Five severity grades: Grade 0–4
- 🎯 Prediction confidence and class probabilities
- 🔍 Grad-CAM explainability
- ⚡ FastAPI inference API
- 🖥️ Interactive Streamlit application
- 🐳 Docker containerization
- 🔗 Docker Compose for API + Streamlit
- 📦 Hugging Face model hosting
- 🔐 SHA256 model integrity verification
- 📈 Prediction monitoring
- 📉 Data/drift monitoring utilities
- 🧪 Automated pytest test suite
- ⚙️ YAML-based configuration
- 🔄 Reproducible inference pipeline

---

## 🏥 Problem Statement

Knee Osteoarthritis is commonly assessed using radiographic imaging and severity grading.

Manual assessment can be time-consuming and may introduce variability between assessments.

This project explores how Deep Learning can assist with automated severity grading by transforming a knee X-ray into:

1. Predicted OA grade
2. Prediction confidence
3. Probability distribution across all grades
4. Visual explanation using Grad-CAM

The system is intended as an **AI-assisted decision-support prototype**, not a replacement for clinical diagnosis.

---

## 🧠 Model

The project uses an **Ordinal ResNet50** architecture.

Instead of treating the five grades as completely unrelated classes, ordinal classification models the ordered nature of OA severity.

### Output

| Grade | Severity |
|---|---|
| Grade 0 | No OA |
| Grade 1 | Mild/early changes |
| Grade 2 | Moderate changes |
| Grade 3 | Advanced changes |
| Grade 4 | Severe changes |

The deployed model is:

```text
Architecture: OrdinalResNet50
Experiment: Experiment 5
Backbone: ResNet50
Input Size: 224 × 224
Number of Classes: 5
Number of Ordinal Thresholds: 4
```
---

## 📊 Model Performance

The recorded evaluation metrics for Experiment 5 are:

| Metric                   |  Score |
| ------------------------ | -----: |
| Accuracy                 | 60.93% |
| Macro F1                 | 59.89% |
| Quadratic Weighted Kappa | 77.07% |

---

## 🔍 Explainability with Grad-CAM

The application includes Grad-CAM visualization to make model predictions more interpretable.

For a prediction, the system can generate:

- Original X-ray
- Grade-specific heatmaps
- Grade-specific overlays

Example output structure:

artifacts/
└── gradcam/
    └── <prediction_id>/
        ├── original.jpg
        ├── prediction.json
        ├── grade_0_heatmap.jpg
        ├── grade_0_overlay.jpg
        ├── grade_1_heatmap.jpg
        ├── grade_1_overlay.jpg
        ├── grade_2_heatmap.jpg
        ├── grade_2_overlay.jpg
        ├── grade_3_heatmap.jpg
        ├── grade_3_overlay.jpg
        ├── grade_4_heatmap.jpg
        └── grade_4_overlay.jpg

---

## 🏗️ System Architecture

                    ┌──────────────────────┐
                    │     Knee X-ray       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Preprocessing  │
                    │ Resize + Normalize   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Ordinal ResNet50    │
                    │  Experiment 5        │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌──────────────────────┐
        │  OA Grade 0–4   │        │ Class Probabilities  │
        └─────────────────┘        └──────────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │    Grad-CAM         │
        │ Explainability      │
        └─────────────────────┘
        
               ┌───────────────┐   
               |   API Layer   |
               └───────────────┘
                       |
           ┌───────────┴───────────┐
           ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐   
    |   FastAPI API   |    |   Streamlit UI  |
    └─────────────────┘    └─────────────────┘
           |                       |
           └───────────┬───────────┘
                       ▼
               ┌─────────────────┐
               |  Docker Compose |
               └─────────────────┘
               
---

## 📁 Project Structure

AI-Knee-Detection-Project/
│
├── api/
│   ├── __init__.py
│   ├── dependencies.py
│   ├── main.py
│   └── schemas.py
│
├── configs/
│   ├── __init__.py
│   ├── config.yaml
│   └── model.yaml
│
├── monitoring/
│   ├── drift.py
│   ├── metrics.py
│   └── monitor.py
│
├── src/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── inference/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── preprocessing.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_inference.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── LICENSE

---

## ⚡ Quick Start

### 1. Clone the repository
git clone https://github.com/Dhritikamal191/AI-Knee-Detection-Project.git
cd AI-Knee-Detection-Project

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the FastAPI service
uvicorn api.main:app --reload

API:
http://localhost:8000

Swagger documentation:
http://localhost:8000/docs

### 🖥️ Run Streamlit
streamlit run app.py

Open:
http://localhost:8501

### 🐳 Docker Deployment
The project supports running both the API and Streamlit application using Docker Compose.

Build
docker compose build

Start
docker compose up

Run in background
docker compose up -d

Check containers
docker compose ps

The project exposes:

FastAPI:
http://localhost:8000

Swagger:
http://localhost:8000/docs

Streamlit:
http://localhost:8501

Stop
docker compose down

---

## 🔌 API

### Health Check

GET /

and:

GET /health

### Prediction

POST /predict

The endpoint accepts a knee X-ray image and returns the model prediction.

Example response structure:

{
    "predicted_grade": 2,
    "confidence": 84.28,
    "grade_probabilities": {
        "Grade 0": 0.64,
        "Grade 1": 1.54,
        "Grade 2": 84.28,
        "Grade 3": 13.52,
        "Grade 4": 0.03
    }
}

The exact values depend on the uploaded image.

---

## 🔐 Model Version Verification

The deployed model is hosted on Hugging Face and downloaded during inference.

The application calculates a SHA256 checksum for the downloaded model.

Current deployed model checksum:

f2200b43966dce1498e47ad6ee45cb35e5cec831246b7667323e16fd9d7e1667

This provides an additional mechanism for verifying that the loaded model corresponds to the expected model artifact.

---

## ⚙️ Configuration

Model configuration is maintained in:

configs/model.yaml

Important configuration parameters include:

model:
  name: ordinal_resnet50
  experiment: experiment5
  architecture: OrdinalResNet50


  input:
    image_size: 224
    channels: 3


  output:
    num_classes: 5

Preprocessing configuration includes ImageNet normalization:

normalize:
  mean:
    - 0.485
    - 0.456
    - 0.406


  std:
    - 0.229
    - 0.224
    - 0.225

Keeping these settings outside the application code makes the inference pipeline easier to maintain and reproduce.

---

## 📈 Monitoring

The project includes monitoring utilities for:

- Prediction metrics
- Model monitoring
- Data drift detection

Located in:

monitoring/
├── metrics.py
├── monitor.py
└── drift.py

These components provide a foundation for monitoring model behavior after deployment.

---

## 🧪 Testing

The project includes automated tests covering:

-API endpoints
- Invalid file handling
- Model creation
- Model output shape
- Prediction output
- Prediction grade validity
- Confidence validity
- Grade probability validity
- Probability normalization
- Preprocessing

Run:

python -m pytest -v tests/

Current test suite:

11 passed

---

## 🛠️ Technology Stack

### Machine Learning
- Python
- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- SciPy
  
### Computer Vision
- OpenCV
- Pillow
- Albumentations
- Grad-CAM
  
### API & Application
- FastAPI
- Uvicorn
- Streamlit
- Pydantic
  
### MLOps / Engineering
- Docker
- Docker Compose
- Pytest
- YAML configuration
- Model SHA256 verification
- Monitoring utilities
- MLflow

### Model Hosting
- Streamlit

---

## 💼 Business / Real-World Value

This project demonstrates how a computer vision model can be transformed from a research experiment into a deployable ML application.

The system addresses several practical ML engineering requirements:

### 1. Automated prediction

Reduces the need for manually processing every image through a classification workflow.

### 2. Ordinal reasoning

The model accounts for the ordered nature of OA severity rather than treating grades as completely unrelated categories.

### 3. Explainability

Grad-CAM provides visual insight into the image regions influencing model predictions.

### 4. Production API

FastAPI exposes the trained model as a reusable inference service.

### 5. User-facing application

Streamlit provides a simple interface for interacting with the model without requiring programming knowledge.

### 6. Deployment

Docker and Docker Compose make the application easier to reproduce across environments.

### 7. Monitoring

Monitoring utilities provide a foundation for detecting changes in model behavior and input data after deployment.

### 8. Reproducibility

Configuration files, model version identification and SHA256 verification help maintain consistency between model development and deployment.

---

## ⚠️ Medical Disclaimer

This project is intended for educational, research and AI engineering purposes.

It is not a certified medical device and should not be used as a standalone diagnostic system.

Predictions should not replace assessment by qualified healthcare professionals.

---

## 🔮 Future Improvements

Potential future development includes:

-Larger and more diverse datasets
- External validation
- Improved class imbalance handling
- Calibration of confidence scores
- Advanced model monitoring
- Automated retraining pipelines
- CI/CD deployment
- More comprehensive model evaluation
- Additional explainability techniques
- Clinical validation
- Secure production API deployment

---

## 👨‍💻 Author

Dhritikamal Das

MSc MACS | Data & AI / Machine Learning

Portfolio: https://dhritikamal191.github.io

---

## ⭐ Project Highlights

Deep Learning
      ↓
Ordinal Classification
      ↓
Computer Vision
      ↓
Grad-CAM Explainability
      ↓
FastAPI
      ↓
Streamlit
      ↓
Docker
      ↓
Monitoring
      ↓
Automated Testing
      ↓
Production-oriented ML System

If you find the project useful, consider giving the repository a ⭐.
