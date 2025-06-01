import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Set page title
st.set_page_config(page_title="Crop Disease Detection", layout="centered")
st.title("🌿 Crop Disease Detection")
st.markdown("Upload a leaf image to predict the disease class.")

# Load the model
MODEL_PATH = 'model/crop_disease_model.h5'
model = load_model(MODEL_PATH)

# Automatically get class names from dataset folder
DATASET_DIR = 'PlantVillage'
class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)

    if predicted_index < len(class_names):
        predicted_class = class_names[predicted_index]
        st.success(f"✅ **Predicted Class:** {predicted_class}")
        st.info(f"📊 Confidence: {prediction[0][predicted_index]*100:.2f}%")
    else:
        st.warning("⚠️ Prediction index out of class range.")
