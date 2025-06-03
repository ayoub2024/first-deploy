import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Charger le modèle
@st.cache_resource
def load_my_model():
    return load_model("modele.keras")  # Modifie le chemin selon l'emplacement de ton modèle

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
    # Afficher l'image uploadée
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    # Conversion en array pour OpenCV
    image_array = np.array(image)
    image_resized = cv2.resize(image_array, (224, 224))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)

    if st.button("Lancer la prédiction"):
        predictions = model.predict(image_input)
        confidence_score = np.max(predictions)
        predicted_class = np.argmax(predictions)

        if confidence_score < 0.6:
            st.warning("Ce n'est pas un billet Djiboutien (confiance faible)")
        else:
            st.success(f"Classe prédite : {class_names[predicted_class]}\n\nConfiance : {confidence_score:.2%}")