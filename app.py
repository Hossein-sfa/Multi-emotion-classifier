import streamlit as st
from PIL import Image
import requests
import io

REGISTER_URL = "http://127.0.0.1:8000/register/"
VERIFY_URL = "http://127.0.0.1:8000/verify/"
EMOTION_URL = "http://127.0.0.1:8000/emotion/"

st.title("ML Project\nHossein Safaei 400243050\n\nFace Verification using FastAPI & Streamlit")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Face Verification", "Emotion Detection"])

if page == "Face Verification":
    st.title("Face Verification\nHossein Safaei 400243050")
    mode = st.radio("Choose Mode:", ("Register", "Verify"))
    option = st.radio("Choose Image Input Method:", ("Use Camera", "Upload Image"))
    image = None
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
    elif option == "Use Camera":
        image = st.camera_input("Take a picture")
        if image:
            image = Image.open(image)
    if image:
        st.image(image, caption="Selected Image", use_container_width=True)
        img_byte_arr = io.BytesIO()         # Convert PIL image to bytes
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        if mode == "Register":
            with st.spinner("Registering..."):
                response = requests.post(REGISTER_URL, files={"file": img_byte_arr})
            if response.status_code == 200:
                st.success("Registered successfully!")
            else:
                st.error("Error: Registration failed")
        elif mode == "Verify":
            with st.spinner("Verifying..."):
                response = requests.post(VERIFY_URL, files={"file": img_byte_arr})
            if response.status_code == 200:
                result = response.json()
                if result["is_match"]:
                    st.success(f"Match found! Distance: {result['euclidean_distance']:.4f}")
                else:
                    st.warning(f"No match found! Distance: {result['euclidean_distance']:.4f}")
            else:
                st.error("Error: Verification failed!")
elif page == "Emotion Detection":
    text_input = st.text_area("Enter a sentence to analyze emotions:")
    if st.button("Analyze Emotions"):
        if text_input:
            with st.spinner("Analyzing..."):
                response = requests.post(EMOTION_URL, json={"text": text_input})
            if response.status_code == 200:
                emotions = response.json()
                if emotions:
                    st.success("Detected Emotions:")
                    for emotion, confidence in emotions.items():
                        st.write(f"**{emotion.capitalize()}**: {confidence:.2f}")
                else:
                    st.warning("No strong emotions detected above the threshold.")
            else:
                st.error("Error: Emotion analysis failed!")
