import streamlit as st
import requests
from PIL import Image
import io
import os

# Hugging Face API settings
API_URL = "https://api-inference.huggingface.co/models/HareeshE/fruit-classifier-model"
API_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)

    # ✅ (1) Print raw response to console (for logs/debug)
    print("RAW RESPONSE:", response.text)

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as e:
        # Return error + raw response
        return {"error": f"Failed to parse JSON: {str(e)}", "raw_response": response.text}

# Streamlit UI
st.title("🍎 Fruit Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    with st.spinner("🔍 Classifying..."):
        result = query(image_bytes)

    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
        
        # ✅ (2) Show raw response in UI
        st.code(result.get("raw_response", "No response received"), language="text")
    else:
        st.success("✅ Prediction complete!")
        st.write(result)
