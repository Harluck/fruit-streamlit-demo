import streamlit as st
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np

st.set_page_config(page_title="Fruit Classifier 🍎", layout="centered")
st.title("🍌🍎 Fruit Classifier App")
st.write("Upload an image of a fruit, and the model will predict its type.")

# Download model from Hugging Face
with st.spinner("Loading model from Hugging Face..."):
    model_path = hf_hub_download(
        repo_id="HareeshE/fruit-classifier-model",
        filename="fruit_classifier_model.h5"
    )
    model = load_model(model_path)

# Upload image
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_names = ["apple", "banana", "orange"]  # Customize if needed
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"✅ Prediction: **{predicted_class}**")
    st.info(f"🔢 Confidence: **{confidence:.2f}%**")
