import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('model.h5')  # Provide the correct path to your model file

# Define class names
class_names = ['Early Blight', 'Late Blight', 'Normal Health Plant']

# Streamlit app layout
st.title(":potato: Potato Disease Prediction")

st.subheader(f"Upload a potato :potato: leaf :leaves: image to predict the disease.")

uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = np.array(image)
    target_size = (256, 256)
    if image.shape[:2] != target_size:
        image = tf.image.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = round(100 * np.max(predictions[0]), 2)

    st.header(f"Predicted Class: {predicted_class} with Confidence: {confidence}%")
   
