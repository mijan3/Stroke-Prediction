# 🧠 Stroke Prediction System

A full-stack **Stroke Prediction** project that predicts the likelihood of a stroke based on patient health data. This project focuses on **machine learning algorithms implemented from scratch (without using `sklearn` built-in models)** and integrates them into a **backend API** and a **frontend web application**.

---

## 📌 Project Overview

Stroke is one of the leading causes of death and long-term disability worldwide. Early prediction can help in prevention and timely medical intervention. This project aims to:

- Implement core machine learning algorithms **from scratch**
- Train and evaluate models on real stroke-related data
- Deploy the trained model through a backend service
- Provide a user-friendly frontend interface for prediction

---

## 🚀 Features

- ✅ Machine Learning models built **from scratch** (no `sklearn` models)
- ✅ Multiple algorithms implemented and compared
- ✅ KNN used as the **main training and prediction pipeline**
- ✅ RESTful backend for model inference
- ✅ Frontend UI for user input and result visualization
- ✅ Clean project structure for academic and real-world use

---

## 🧠 Implemented Algorithms (From Scratch)

The following algorithms were implemented manually using only **NumPy, Pandas, and core Python**:

- **K-Nearest Neighbors (KNN)** ✅ *(Primary Model)*
- **Random Forest**
- **XGBoost (Gradient Boosting - Custom Implementation)**
- **Linear Discriminant Analysis (LDA)**
- **Support Vector Machine (SVM)**

> ⚠️ No `sklearn` built-in models were used for training or prediction.

---

## 🏗️ System Architecture

```
Stroke-Prediction/
│
├── backend/
│   ├── app.py                 # Backend API (model inference)
│   ├── training.py            # Model training pipeline (KNN main)
│   ├── stroke-data.csv        # Dataset
│   └── stroke_prediction.pkl  # Saved trained model
│
├── frontend/
│   ├── src/                   # React source code
│   ├── public/
│   ├── package.json
│   └── README.md
│
├── notebook/
│   └── Stroke_Prediction.ipynb # Algorithm development & experiments
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Technologies Used

### 📊 Machine Learning
- Python
- NumPy
- Pandas
- Custom ML implementations

### 🔙 Backend
- Flask
- Python REST API
- Model serialization (joblib)

### 🎨 Frontend
- React.js
- HTML / CSS / JavaScript
- Fetch API / Axios

---

## 📂 Dataset

- **Source**: Stroke prediction dataset
- **Format**: CSV
- **Target Variable**: `stroke`
- **Features include**:
  - Age
  - Gender
  - Hypertension
  - Heart disease
  - BMI
  - Smoking status
  - Glucose level

---

## 🧪 Model Training Pipeline

- Data Cleaning & Preprocessing
- Feature Encoding & Scaling
- Train/Test Split
- **KNN selected as the main pipeline model**
- Model evaluation using accuracy and confusion matrix
- Model saved for backend inference

---

## ▶️ How to Run the Project

### 1️⃣ Backend Setup

```bash
cd backend
conda create -n strokeprc python=3.8 -y
conda activate strokeprc
pip install -r requirements.txt
python app.py
```

Backend will start at:
```
http://localhost:5000
```

---

### 2️⃣ Frontend Setup

```bash
cd frontend
npm install
npm start
```

Frontend will start at:
```
http://localhost:3000
```

---

## 🔌 API Endpoint Example

**POST** `/predict`

```json
{
  "age": 45,
  "hypertension": 0,
  "heart_disease": 1,
  "avg_glucose_level": 120.5,
  "bmi": 28.3
}
```

**Response**:
```json
{
  "stroke_prediction": 0
}
```

---

## 📈 Results

- KNN achieved reliable accuracy and was selected for deployment
- Scratch implementations helped in deep understanding of algorithms
- The system successfully predicts stroke risk from user input

---

## 🎯 Learning Outcomes

- Implemented ML algorithms from scratch
- Built a complete ML pipeline without sklearn models
- Integrated ML models into real-world web applications
- Improved understanding of data preprocessing and evaluation

---

## 👨‍💻 Author

**Mijanur Rahman**  
CSE Student  
North South University

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

⭐ If you find this project useful, feel free to give it a star!

