import streamlit as st
import numpy as np
from PIL import Image
import gdown
from tensorflow.keras.models import load_model
import os

MODEL_URL = 'https://drive.google.com/file/d/1-0uFi1cJtt_ZF5vX7CsMj1DO5MyT2WzT/view?usp=drive_link'
MODEL_PATH = 'modele.keras'

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle depuis Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_my_model()

class_names = [
    'Ceci est une billet de 1000 franc',
    'Ceci est une billet de 10000 franc',
    'Ceci est une billet de 2000 franc',
    'Ceci est une billet de 5000 franc',
    'Ceci est une pieces de 10 franc',
    'Ceci est une pieces de 100 franc',
    'Ceci est une pieces de 20 franc',
    'Ceci est une pieces de 250 franc',
    'Ceci est une pieces de 50 franc',
    'Ceci est une pieces de 500 franc'
]

st.title("Reconnaissance de billets/pièces djiboutiens")

uploaded_file = st.file_uploader("Choisis une image à analyser", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_input = np.expand_dims(image_array, axis=0)

    if st.button("Lancer la prédiction"):
        predictions = model.predict(image_input)
        confidence_score = np.max(predictions)
        predicted_class = np.argmax(predictions)
        if confidence_score < 0.6:
            st.warning("Ce n'est pas un billet Djiboutien (confiance faible)")
        else:
            st.success(f"Classe prédite : {class_names[predicted_class]}\n\nConfiance : {confidence_score:.2%}")
