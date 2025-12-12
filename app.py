import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import zipfile
import requests
from io import BytesIO
import shutil

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Title and description
st.title("🌿 Plant Disease Detection System")
st.write("Upload an image of a plant leaf to detect diseases using deep learning CNN model")

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

# Function to download and extract sample images from GitHub
@st.cache_resource
def download_sample_images(github_url):
    """Download zip file from GitHub and extract sample images"""
    try:
        response = requests.get(github_url, timeout=30)
        response.raise_for_status()

        # Extract to temporary directory
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall('sample_images')

        return True, "Sample images loaded successfully"
    except Exception as e:
        return False, f"Error loading samples: {str(e)}"

# Function to organize sample images by disease class
def get_sample_images_by_class():
    """Get sample images organized by disease class"""
    sample_dict = {}
    sample_dir = 'sample_images'

    if os.path.exists(sample_dir):
        for class_name in class_names:
            class_path = os.path.join(sample_dir, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    sample_dict[class_name] = [
                        os.path.join(class_path, img) for img in images[:3]
                    ]

    return sample_dict

# Load model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('plant_disease_model.h5')
        return model
    except:
        return None

# Preprocess image
def preprocess_image(image):
    """Preprocess image for model prediction"""
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image

    # Resize to 150x150
    img = img.resize((150, 150))
    # Convert to array
    img_array = np.array(img)
    # Normalize
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make prediction
def predict(model, image):
    """Make prediction on image"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence, predictions[0]

# Sidebar - GitHub URL input and sample images
with st.sidebar:
    st.header("📁 Sample Images Configuration")

    github_url = st.text_input(
        "GitHub ZIP URL",
        value="https://github.com/YOUR_USERNAME/plant-disease-samples/archive/refs/heads/main.zip",
        help="Paste your GitHub zip file URL containing sample images"
    )

    if st.button("📥 Load Sample Images"):
        with st.spinner("Downloading and extracting samples..."):
            success, message = download_sample_images(github_url)
            if success:
                st.success(message)
            else:
                st.warning(message)

    st.divider()

    st.header("About")
    st.write(
        "This CNN-based application detects plant diseases from leaf images. "
        "Trained on PlantVillage dataset with 15 disease categories."
    )
    st.write("**Model Architecture:**")
    st.write("- Conv2D (32 filters) → MaxPooling")
    st.write("- Conv2D (64 filters) → MaxPooling")
    st.write("- Flatten → Dense (32) → Softmax (15 classes)")
    st.write("- Input: 150×150 RGB | Output: 15 classes")

    st.header("Supported Plants")
    st.write("🍎 Apple | 🌽 Corn | 🍇 Grape | 🥔 Potato")

# Main content area
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Sample Images", "ℹ️ Info"])

with tab1:
    st.header("Upload & Predict")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image", 
            type=["jpg", "jpeg", "png"],
            key="main_upload"
        )

    with col2:
        st.subheader("Or Use Sample Image")
        sample_images_dict = get_sample_images_by_class()
        if sample_images_dict:
            selected_class = st.selectbox(
                "Select a disease class",
                options=list(sample_images_dict.keys()),
                format_func=lambda x: x.replace('_', ' ')
            )

            if selected_class and sample_images_dict[selected_class]:
                sample_image_path = sample_images_dict[selected_class][0]
                uploaded_file = sample_image_path
                st.info(f"Selected sample: {selected_class.replace('_', ' ')}")

    if uploaded_file is not None:
        model = load_model()

        if model is not None:
            # Display and process image
            image = Image.open(uploaded_file) if isinstance(uploaded_file, str) else Image.open(uploaded_file)

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.image(image, use_container_width=True, caption="Input Image")

            with result_col2:
                with st.spinner('🔬 Analyzing image...'):
                    predicted_class, confidence, all_predictions = predict(model, image)

                st.subheader("📋 Prediction Results")

                # Format class name
                plant_type, disease = predicted_class.split('___')

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Plant", plant_type.replace('_', ' '))
                with col_b:
                    st.metric("Status", disease.replace('_', ' '))

                st.metric("Confidence Score", f"{confidence * 100:.2f}%")
                st.progress(float(confidence))

                # Color coding based on confidence
                if confidence > 0.8:
                    st.success("✅ High confidence prediction")
                elif confidence > 0.6:
                    st.info("ℹ️ Moderate confidence - verify with expert")
                else:
                    st.warning("⚠️ Low confidence - manual verification recommended")

            # Top 3 predictions
            st.subheader("🔝 Top 3 Predictions")
            top_3_idx = np.argsort(all_predictions)[-3:][::-1]

            cols = st.columns(3)
            for i, idx in enumerate(top_3_idx):
                with cols[i]:
                    plant, disease = class_names[idx].split('___')
                    conf = all_predictions[idx] * 100
                    st.metric(
                        plant.replace('_', ' '),
                        f"{disease.replace('_', ' ')}",
                        f"{conf:.1f}%"
                    )

            # Recommendations
            if 'healthy' not in predicted_class.lower() and confidence > 0.7:
                st.subheader("⚠️ Disease Detected - Recommendations")
                st.warning(
                    "Your plant has been identified with a disease. Consider:\n\n"
                    "✓ Consult local agricultural extension office\n"
                    "✓ Remove affected leaves immediately\n"
                    "✓ Apply appropriate fungicide or treatment\n"
                    "✓ Improve air circulation and reduce humidity\n"
                    "✓ Isolate plant from others if contagious"
                )
            elif 'healthy' in predicted_class.lower():
                st.success("✅ Your plant appears healthy! Continue good care practices.")
        else:
            st.error("❌ Model not found. Ensure 'plant_disease_model.h5' exists.")

with tab2:
    st.header("📁 Sample Images by Disease Class")

    sample_images_dict = get_sample_images_by_class()

    if sample_images_dict:
        st.info(f"✅ Found samples for {len(sample_images_dict)} disease classes")

        # Display file structure
        st.subheader("📂 File Structure")

        file_structure = "```"
        file_structure += "\nsample_images/\n"
        for class_name in sorted(sample_images_dict.keys()):
            plant_type, disease = class_name.split('___')
            file_structure += f"├── {class_name}/\n"
            for img_path in sample_images_dict[class_name][:2]:
                file_structure += f"│   └── {os.path.basename(img_path)}\n"
        file_structure += "```"

        st.markdown(file_structure)

        st.divider()

        # Display images by class
        st.subheader("🖼️ Sample Images Gallery")

        for class_name in sorted(sample_images_dict.keys()):
            plant_type, disease = class_name.split('___')
            disease_name = f"{plant_type.replace('_', ' ')} - {disease.replace('_', ' ')}"

            with st.expander(f"📌 {disease_name}", expanded=False):
                image_cols = st.columns(3)

                for idx, img_path in enumerate(sample_images_dict[class_name]):
                    with image_cols[idx % 3]:
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                            st.caption(os.path.basename(img_path))
                        except Exception as e:
                            st.error(f"Error loading: {e}")
    else:
        st.warning(
            "❌ No sample images found. Please:\n\n"
            "1. Create a GitHub repository with sample images organized by disease class\n"
            "2. Create a zip file of the folder structure\n"
            "3. Paste the GitHub raw zip URL in the sidebar\n"
            "4. Click 'Load Sample Images'"
        )

        st.subheader("📋 Expected Folder Structure")
        st.code("""
sample_images/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
├── Apple___Black_rot/
│   ├── image1.jpg
│   └── ...
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
├── Corn_(maize)___Common_rust_/
├── Corn_(maize)___Northern_Leaf_Blight/
├── Corn_(maize)___healthy/
├── Grape___Black_rot/
├── Grape___Esca_(Black_Measles)/
├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
├── Grape___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
└── Potato___healthy/
        """)

        st.subheader("🔧 Setup Instructions")
        st.markdown("""
1. **Download sample images** from [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
2. **Create GitHub repo** with folder structure matching disease classes
3. **Create ZIP** of the folder and push to GitHub
4. **Get raw ZIP URL**: `https://github.com/USER/REPO/archive/refs/heads/main.zip`
5. **Paste URL** in sidebar and click "Load Sample Images"
        """)

with tab3:
    st.header("ℹ️ Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏗️ Model Details")
        st.write("""
**Architecture:**
- Input Layer: 150×150×3 RGB images
- Conv2D Layer 1: 32 filters, 3×3 kernel, ReLU
- MaxPooling2D: 2×2 pool
- Conv2D Layer 2: 64 filters, 3×3 kernel, ReLU
- MaxPooling2D: 2×2 pool
- Flatten Layer
- Dense Layer: 32 units, ReLU
- Output Layer: 15 units, Softmax

**Performance:**
- Training Accuracy: ~98.7%
- Validation Accuracy: ~98.6%
- Loss: ~0.04
        """)

    with col2:
        st.subheader("🌱 Supported Diseases")

        diseases_by_plant = {}
        for class_name in class_names:
            plant, disease = class_name.split('___')
            if plant not in diseases_by_plant:
                diseases_by_plant[plant] = []
            diseases_by_plant[plant].append(disease)

        for plant in sorted(diseases_by_plant.keys()):
            with st.expander(f"🌾 {plant.replace('_', ' ')}", expanded=True):
                for disease in diseases_by_plant[plant]:
                    st.write(f"• {disease.replace('_', ' ')}")

    st.divider()

    st.subheader("📊 Dataset Information")
    st.write("""
**PlantVillage Dataset:**
- 15 disease classes
- 4 plant types: Apple, Corn, Grape, Potato
- 20,638+ training images
- 150×150 pixel resolution
- Publicly available: [PlantVillage GitHub](https://github.com/spMohanty/PlantVillage-Dataset)
    """)

    st.subheader("✅ Image Requirements")
    st.write("""
For best results:
- ✓ Clear, well-lit leaf images
- ✓ Single leaf or small portion
- ✓ Minimize background
- ✓ No motion blur
- ✓ JPG, JPEG, or PNG format
- ✓ Recommended: 150×150 or larger
    """)
