import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
try:
    model = tf.keras.models.load_model('pneumonia_cnn_model.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Streamlit app
st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to check for pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    try:
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        image = image.resize((150, 150))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize pixel values
        image = image.reshape(1, 150, 150, 1)  # Reshape for model input

        # Make a prediction
        prediction = model.predict(image)
        result = "Pneumonia" if prediction > 0.5 else "Normal"
        st.write(f"**Prediction:** {result}")
    except Exception as e:
        st.error(f"Error processing the image: {e}")