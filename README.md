 🏎️ F1 Podium Prediction (Machine Learning)

#📌 Overview

This project predicts the **top 3 finishers (podium)** of a Formula 1 race using Machine Learning.

Instead of directly predicting the winner, it uses a **ranking-based regression model**:

* Each driver gets a score
* Drivers are sorted based on score
* Top 3 = predicted podium

---

## 🚀 Features

* Predict podium using **circuit name** (e.g., monaco, bahrain)
* Automatic feature generation from past races
* FastAPI-based backend
* Uses real-world F1 dataset

---

## 🧠 ML Approach

* Model: Random Forest Regressor
* Type: Regression + Ranking
* Key Features:

  * Grid position
  * Driver recent performance
  * Constructor performance

---

## ⚙️ How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python -m src.train
```

### 3. Run the API

```
uvicorn src.app:app --reload
```

### 4. Open in browser

```
http://127.0.0.1:8000/docs
```

---

## 🌐 Example Usage

Input:

```
/predict/monaco
```

Output:

```
[
  {"name": "Driver 1", "pred_score": 0.91},
  {"name": "Driver 2", "pred_score": 0.88},
  {"name": "Driver 3", "pred_score": 0.85}
]
```

---

## 📁 Project Structure

```
src/
  preprocess.py
  train.py
  predict.py
  app.py
```

---

## ⚠️ Note

The trained model file is not included due to size limits.

To generate it:

```
python -m src.train
```

---

## 👨‍💻 Author 
JAGAN

http://127.0.0.1:8000/

