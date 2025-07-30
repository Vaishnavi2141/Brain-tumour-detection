import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

st.title("🧠 Brain Tumor Detection")
st.markdown("Upload an MRI image to predict whether it has a brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(image)
    img = cv2.resize(img_array, (64, 64)) / 255.0
    img = img.reshape(1, 64, 64, 3)

    # Predict
    prediction = model.predict(img)
    result = "🧠 Tumor Detected" if prediction[0][0] > 0.5 else "✅ No Tumor Detected"

    st.subheader("Prediction:")
    st.success(result)
Add Streamlit app for deployment

