# Policy Draft Comment Analysis System

**An AI-powered platform for analyzing public sentiment, stance, and policy topics from large-scale text data.**

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-red)

## 📌 Overview
This project processes public comments on government policy drafts (e.g., "Waste-to-Energy Plant", "Metro Line") to provide actionable insights for policymakers. It utilizes a **Hybrid Sentiment Analysis Engine** combining DistilBERT with rule-based logic to handle complex linguistic nuances like mixed sentiment, conditional support, and policy-specific terminology.

## ✨ Key Features
*   **Hybrid Inference Engine**: Combines Deep Learning (DistilBERT) with linguistically aware rules to detect "Mixed" sentiment, sarcasm, and conditional approval.
*   **Three-Pillar Analysis**:
    *   **Sentiment**: Positive, Negative, Neutral, Mixed, Invalid.
    *   **Stance**: Support, Oppose, Suggestion, Query.
    *   **Topic**: Automatically categorizes comments into policy domains.
*   **Interactive Web UI**: A modern, glassmorphism-styled interface for real-time analysis.
*   **Analytics Dashboard**: A Streamlit dashboard for aggregate statistics and data exploration.

## 📂 Project Structure
```
policy_analysis/
├── api/                    # FastAPI Backend & Endpoints
├── web/                    # Frontend (HTML/JS/CSS)
├── src/                    # Core ML Logic (Inference, Training, Preprocessing)
├── dashboard/              # Streamlit Analytics Dashboard
├── models_improved/        # Trained Model Artifacts (Generated locally)
├── data/                   # Dataset (Included)
└── run_hybrid_inference.py # Main Hybrid Inference Script
```

## 🚀 Getting Started

### Prerequisites
*   Python 3.8 or higher
*   pip

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone <YOUR_REPO_URL>
cd Policy-comments-analyser
pip install -r policy_analysis/requirements.txt
```

### 2. Build the Model (Crucial Step)
To keep the repository light, the large model files are **not** included. You must generate them locally using the provided training script and dataset:

```bash
python policy_analysis/train_enhanced_sentiment.py
```
*   **Time**: ~5-10 minutes on GPU, ~20+ minutes on CPU.
*   **Output**: Saves the model to `policy_analysis/models_improved/`.

### 3. Run the Web Interface
Start the backend server to launch the application:

```bash
python -m uvicorn policy_analysis.api.main:app --reload
```
👉 **Open your browser at:** [http://127.0.0.1:8000/app.html](http://127.0.0.1:8000/app.html)

### 4. (Optional) Run the Dashboard
To see aggregate analytics of the entire dataset:
```bash
streamlit run policy_analysis/dashboard/app.py
```

## 🧠 Model Details
The system uses a **Hybrid Approach**:
1.  **Base Model**: `distilbert-base-uncased` fine-tuned on 50k policy comments.
2.  **Logic Layer**: A post-processing layer that handles:
    *   **Mixed Sentiment**: Detects structure like *"Good but..."* or *"Support if..."*.
    *   **Invalid Input**: Filters gibberish or too-short text.
    *   **Domain Lexicons**: Recognizes policy-specific terms (e.g., "bureaucracy", "compliance").

## 📊 Dataset
The dataset consists of **50,000+ public comments** scraped from various public policy consultation portals. It includes labeled fields for sentiment, stance, and policy categories.

