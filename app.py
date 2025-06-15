import streamlit as st
import requests
from PIL import Image
import io
import os

# Hugging Face API settings
API_URL = "https://api-inference.huggingface.co/models/HareeshE/fruit-classifier-model"
API_TOKEN = os.getenv("HF_TOKEN")

headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

st.title("🍉 Fruit Image Classifier")
st.write("Upload an image of a fruit and get the prediction.")

uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    with st.spinner("Classifying..."):
        response = requests.post(API_URL, headers=headers, data=img_bytes)

    try:
        output = response.json()
        prediction = output[0]["label"]
        confidence = output[0]["score"] * 100
        st.success(f"✅ Prediction: {prediction} ({confidence:.2f}%)")
    except Exception as e:
        st.error("❌ Error processing prediction. Check model deployment or API token.")
