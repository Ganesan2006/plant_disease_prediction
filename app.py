import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Title and description
st.title("🌿 Plant Disease Detection")
st.write("Upload an image of a plant leaf to detect diseases using deep learning")

# Class names - 15 disease categories from PlantVillage dataset
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy'
]

# Load model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('plant_disease_model.keras')
        return model
    except:
        st.error("Model file not found. Please ensure 'plant_disease_model.keras' is in the same directory.")
        return None

# Preprocess image
def preprocess_image(image):
    # Resize to 150x150 as per model training
    img = image.resize((150, 150))
    # Convert to array
    img_array = np.array(img)
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make prediction
def predict(model, image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence, predictions[0]

# Main app
model = load_model()

if model is not None:
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        # Make prediction
        with st.spinner('Analyzing image...'):
            predicted_class, confidence, all_predictions = predict(model, image)

        with col2:
            st.subheader("Prediction Results")

            # Format class name for better readability
            plant_type, disease = predicted_class.split('___')

            st.metric("Plant Type", plant_type.replace('_', ' '))
            st.metric("Disease Status", disease.replace('_', ' '))
            st.metric("Confidence", f"{confidence * 100:.2f}%")

            # Confidence bar
            st.progress(float(confidence))

        # Show top 3 predictions
        st.subheader("Top 3 Predictions")
        top_3_idx = np.argsort(all_predictions)[-3:][::-1]

        for idx in top_3_idx:
            plant, disease = class_names[idx].split('___')
            conf = all_predictions[idx] * 100
            st.write(f"**{plant.replace('_', ' ')} - {disease.replace('_', ' ')}**: {conf:.2f}%")

        # Recommendations based on disease
        if 'healthy' not in predicted_class.lower() and confidence > 0.7:
            st.subheader("⚠️ Recommendations")
            st.info(
                "Your plant appears to have a disease. Consider:\n"
                "- Consulting with a local agricultural expert\n"
                "- Removing affected leaves if possible\n"
                "- Applying appropriate fungicides or treatments\n"
                "- Improving air circulation around plants"
            )
        elif 'healthy' in predicted_class.lower():
            st.success("✅ Your plant appears healthy! Keep up the good care.")
else:
    st.warning("Please ensure the model file 'plant_disease_model.h5' is available.")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.write(
        "This application uses a Convolutional Neural Network (CNN) "
        "trained on the PlantVillage dataset to detect plant diseases."
    )
    st.write("**Model Architecture:**")
    st.write("- 2 Convolutional layers")
    st.write("- 2 MaxPooling layers")
    st.write("- Dense layers with softmax")
    st.write("- Input: 150x150 RGB images")
    st.write("- Output: 15 disease classes")

    st.header("Supported Plants")
    st.write("- 🍎 Apple")
    st.write("- 🌽 Corn (Maize)")
    st.write("- 🍇 Grape")
    st.write("- 🥔 Potato")

    st.header("Tips for Best Results")
    st.write("- Use clear, well-lit images")
    st.write("- Focus on the leaf area")
    st.write("- Avoid blurry images")
    st.write("- Single leaf works best")
