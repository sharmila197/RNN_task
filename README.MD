# 🏃 Human Activity Recognition using RNN

A Machine Learning project that uses a **Recurrent Neural Network (RNN)** to recognize human activities from sensor-based data.

The trained model is integrated with a **Streamlit web application**, where users can upload a CSV dataset and get activity predictions with confidence scores.

---

## 📌 Project Overview

Human Activity Recognition (HAR) is the process of identifying human activities using sensor data.

In this project, an RNN model is trained using **561 numerical features** extracted from sensor measurements.

The model can classify the following activities:

- 🚶 WALKING
- ⬆️ WALKING_UPSTAIRS
- ⬇️ WALKING_DOWNSTAIRS
- 🪑 SITTING
- 🧍 STANDING
- 🛌 LAYING

---

## 🎯 Objectives

- Preprocess sensor-based activity data
- Scale the input features
- Train an RNN deep learning model
- Evaluate the trained model
- Save the trained model and preprocessing objects
- Build a Streamlit web application
- Upload CSV data through the web application
- Predict human activities
- Display prediction confidence
- Visualize activity distribution
- Download prediction results

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- Matplotlib
- Pickle
- Jupyter Notebook

---

## 📂 Project Structure

```text
RNN/
│
├── rnn.ipynb/
│   ├── rnn.ipynb
│   ├── rnn_model.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── train.csv/
│   └── train.csv
│
├── test.csv/
│   └── test.csv
│
├── app.py
└── README.md