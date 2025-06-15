import streamlit as st
import requests
from PIL import Image
import io
import os

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/HareeshE/fruit-classifier-model"
API_TOKEN = os.getenv("HF_TOKEN")  # make sure to add this token in Streamlit Secrets
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"error": "⚠️ Failed to decode response. Check model status or input format."}

st.title("🍎 Fruit Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    with st.spinner("🔍 Classifying..."):
        result = query(image_bytes)

    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
    else:
        st.success("✅ Prediction complete!")
        st.json(result)
