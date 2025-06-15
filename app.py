import streamlit as st
import requests
from PIL import Image
import io
import os

# Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/HareeshE/fruit-classifier-model"
API_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

st.title("Fruit Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    with st.spinner("Classifying..."):
        result = query({"inputs": image_bytes})

    if "error" in result:
        st.error("Error processing prediction. Check model deployment or API token.")
    else:
        st.success("Prediction complete!")
        st.write(result)
