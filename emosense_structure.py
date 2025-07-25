# EmoSense - Emotion Detection ML Project Structure 🌹

EmoSense/
├── app.py                  # Streamlit/Gradio frontend for interaction
├── train.py                # Model training script
├── predict.py              # Prediction and inference functions
├── requirements.txt        # All dependencies
├── README.md               # Project overview
├── emotions.csv            # Dataset (or placeholder)
├── model/
│   ├── model.pkl           # Trained model file (for sklearn)
│   └── tokenizer.pkl       # If using custom tokenizer
├── utils/
│   ├── preprocess.py       # Text cleaning & preparation
│   └── labels.json         # Emotion label mappings
└── notebooks/
    └── exploration.ipynb   # Jupyter notebook for EDA & experiments

# app.py (basic Streamlit version)
import streamlit as st
import joblib
from utils.preprocess import clean_text

model = joblib.load("model/model.pkl")
labels = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Fear"}

st.title("🌸 EmoSense: Emotion Classifier")
text = st.text_area("Enter your message:")

if st.button("Predict Emotion"):
    if text:
        clean = clean_text(text)
        prediction = model.predict([clean])[0]
        st.success(f"Detected Emotion: {labels[prediction]}")
    else:
        st.warning("Please enter some text.")

# train.py (basic sklearn model)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("emotions.csv")
X = df["text"]
y = df["label"]

# Vectorize
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
