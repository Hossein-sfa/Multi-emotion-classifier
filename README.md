# Multi-Emotion Classifier using Tiny-BERT

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-✔-green)
![Streamlit](https://img.shields.io/badge/Streamlit-✔-red)
![Tiny-BERT](https://img.shields.io/badge/TinyBERT-✔-purple)

## 🌟 Overview

This project implements a **Multi-Emotion Classifier** using **Tiny-BERT** for sentence classification, **FastAPI** for the backend, and **Streamlit** as the UI. The model predicts **multiple emotions** in text inputs, making it useful for sentiment analysis, chatbots, and social media analytics.

## 🚀 Features
- **Multi-Emotion Classification** based on **Tiny-BERT**
- **FastAPI Backend** for efficient REST API communication
- **Streamlit UI** for an intuitive web-based interface
- **Model Deployment Ready** for both local and cloud environments
- **Supports Emotion Classes:** `praise`, `amusement`, `anger`, `disapproval`, `confusion`, `interest`, `sadness`, `fear`, `joy`, `love`

---

## ⚙️ Tech Stack

- **Model:** [Tiny-BERT](https://huggingface.co/huawei-noah/TinyBERT_General_6L_768D)
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Deep Learning:** PyTorch, Transformers
- **Server:** Uvicorn

---

## 📌 Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/multi-emotion-classifier.git
cd multi-emotion-classifier
```

### **3️⃣ Start the FastAPI Backend**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
- The API will be available at: `http://127.0.0.1:8000`
- Check API Docs at: `http://127.0.0.1:8000/docs`

### **4️⃣ Start the Streamlit Frontend**
```bash
streamlit run app.py
```
- The UI will be available at: `http://localhost:8501`

---

## 📡 API Endpoints

| Method | Endpoint         | Description                   |
|--------|----------------|------------------------------|
| `POST` | `/emotion/`    | Classifies emotions in text |

---

## 🎯 Usage

### **1️⃣ UI Usage (Streamlit)**
1. Run `streamlit run app.py`
2. Navigate to `http://localhost:8501`
3. Enter a sentence and get multi-emotion predictions.

### **2️⃣ API Usage (FastAPI)**
#### **Example Request**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/emotion/' \
  -H 'Content-Type: application/json' \
  -d '{"text": "I am really happy today!"}'
```

#### **Example Response**
```json
{
  "emotions": {
    "joy": 0.95,
    "amusement": 0.72,
    "praise": 0.65,
    "anger": 0.02
  }
}
```

---

## 📊 Model Training (Jupyter Notebook)
The **MultiEmotionClassifier.ipynb** notebook contains:
- **Data Preprocessing**
- **Training Tiny-BERT on Multi-Emotion Dataset**
- **Evaluation Metrics (F1-score, Accuracy, etc.)**
- **Saving the Model for Deployment**

---

## 🛠 Challenges & Innovations
### **Challenges**
- **Tiny-BERT Model Size:** Managing memory efficiency while maintaining accuracy.
- **Real-time API Performance:** Optimizing FastAPI with Uvicorn for speed.
- **Frontend-Backend Communication:** Ensuring seamless data flow between Streamlit and FastAPI.

### **Innovations**
✅ **Quantization Aware Training (QAT)** for model compression.  
✅ **Pipelined FastAPI Requests** using **async I/O** for better performance.  
✅ **Multi-threaded Batch Processing** for concurrent emotion classification.  

---

---

## 🤝 Contributing
We welcome contributions! Feel free to **fork** the repo, create a branch, and submit a **pull request**.

---

## 📬 Contact
- **Author:** Hossein Safaei  
- **Email:** hossaf82@gmial.com  
- **GitHub:** [@Hossein-sfa](https://github.com/Hossein-sfa)

---

### 🚀 **Star this repo if you like it!**

