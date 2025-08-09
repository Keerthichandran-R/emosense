import streamlit as st
import joblib
from utils.predict import predict_emotion, get_sacred_response

# Load trained emotion model
model = joblib.load("model/model.pkl")  # instead of models/emotion_model.pkl


# Streamlit web app
st.set_page_config(page_title="EmoSense 🌿", page_icon="🌸")

st.title("🌿 EmoSense — Emotion-Aware AI")
st.markdown("**A sacred mirror for your emotions.**")

# Input
user_input = st.text_area("📝 What are you feeling right now?", height=150)

# Predict & Display
if st.button("Sense Emotion"):
    if user_input.strip():
        emotion = predict_emotion(model, user_input)
        sacred_message = get_sacred_response(emotion)

        st.success(f"**Emotion Detected:** {emotion.capitalize()}")
        st.info(f"**Sacred Reflection:**\n\n{ sacred_message }")
    else:
        st.warning("Please write something to analyze.")

