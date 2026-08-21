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

## 📊 Model Performance

The recorded evaluation metrics for Experiment 5 are:

| Metric                   |  Score |
| ------------------------ | -----: |
| Accuracy                 | 60.93% |
| Macro F1                 | 59.89% |
| Quadratic Weighted Kappa | 77.07% |
