import re

def clean_text(text):
    return re.sub(r"[^a-zA-Z\s]", "", text.lower())

def predict_emotion(text, model):
    cleaned = clean_text(text)
    return model.predict([cleaned])[0]

def get_sacred_response(emotion):
    responses = {
        "joy": "Your joy is a light. Let it bless the world. 🌞",
        "sadness": "Even your sorrow is sacred. Let the tears flow into healing rivers. 💧",
        "anger": "Your fire can be alchemy. Breathe, transform, rise. 🔥",
        "fear": "You are safe. The unknown is just a sacred path not yet walked. 🌫️",
        "love": "Your heart is open. Let it guide your steps. 💗",
        "neutral": "Stillness is also divine. Listen within. 🌿"
    }
    return responses.get(emotion.lower(), "Your emotion is valid.")
