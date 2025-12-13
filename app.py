import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import zipfile
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ⭐ CRITICAL: Class order MUST match training order
# This order comes directly from image_dataset_from_directory()
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Model path
MODEL_PATH = "plant_disease_model.keras"
ZIP_PATH = "plant_images.zip"
EXTRACT_DIR = ""

# Auto-load sample images from ZIP
@st.cache_resource
def load_sample_images():
    """Extract sample images from ZIP file"""
    if not os.path.exists(ZIP_PATH):
        return None, None

    try:
        # Create extract directory
        os.makedirs(EXTRACT_DIR, exist_ok=True)

        # Extract ZIP
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        # Scan for images
        image_dict = {}
        total_images = 0

        for class_name in CLASS_NAMES:
            class_path = Path(EXTRACT_DIR) / "sample_images" / class_name
            if class_path.exists():
                images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.jpeg"))
                if images:
                    image_dict[class_name] = [str(img) for img in images]
                    total_images += len(images)

        return image_dict, total_images

    except Exception as e:
        st.error(f"Error extracting ZIP: {str(e)}")
        return None, None

# Load model
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess image
def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to 150x150 (training size)
    img = image.resize((150, 150))
    # Convert to array
    img_array = np.array(img)
    # Ensure 3 channels (RGB)
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make prediction
def predict(model, image):
    """Make prediction on image"""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)

    # Get top 5 predictions
    top_indices = np.argsort(predictions[0])[-5:][::-1]
    top_predictions = [
        {
            "class": CLASS_NAMES[idx],
            "confidence": float(predictions[0][idx] * 100)
        }
        for idx in top_indices
    ]

    return top_predictions

# Display file structure
def display_file_structure(image_dict):
    """Display file structure of loaded images"""
    st.markdown("### 📂 Loaded Sample Images")

    for class_name in sorted(image_dict.keys()):
        with st.expander(f"**{class_name}** ({len(image_dict[class_name])} images)"):
            cols = st.columns(4)
            for idx, img_path in enumerate(image_dict[class_name][:20]):  # Show first 20
                with cols[idx % 4]:
                    try:
                        img = Image.open(img_path)
                        st.image(img, caption=Path(img_path).name, use_container_width=True)
                    except:
                        st.error(f"Error loading {img_path}")

# Main app
def main():
    st.title("🌿 Plant Disease Detection")
    st.markdown("---")

    # Load sample images
    with st.spinner("🔄 Loading sample images from local ZIP..."):
        image_dict, total_images = load_sample_images()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")

        if image_dict:
            st.success(f"✅ Classes loaded: {len(image_dict)}")
            st.success(f"📷 Total images: {total_images}")
        else:
            st.warning("⚠️ No sample images loaded")
            st.info("ℹ️ Place 'plant_images.zip' in the app directory to auto-load sample images.")

        st.markdown("---")
        st.markdown("**Model:** CNN (150x150)")
        st.markdown("**Classes:** 15 plant diseases")
        st.markdown("**Accuracy:** ~98%")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "📊 File Structure", "🖼️ Image Gallery", "ℹ️ Info"])

    # Tab 1: Prediction
    with tab1:
        st.header("Upload or Select Image")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

        with col2:
            st.subheader("📁 Select Sample Image")
            if image_dict:
                selected_class = st.selectbox("Select Disease Class", options=sorted(image_dict.keys()))
                if selected_class:
                    selected_image = st.selectbox("Select Image", options=image_dict[selected_class])
            else:
                st.info("No sample images available")
                selected_image = None

        # Choose prediction source
        predict_source = None
        if uploaded_file:
            predict_source = Image.open(uploaded_file)
            st.image(predict_source, caption="Uploaded Image", use_container_width=True)
        elif image_dict and selected_image:
            predict_source = Image.open(selected_image)
            st.image(predict_source, caption="Selected Sample", use_container_width=True)

        # Predict button
        if st.button("🔬 Predict Disease", type="primary", use_container_width=True):
            if predict_source is None:
                st.warning("⚠️ Please upload or select an image first")
            else:
                model = load_model()
                if model is None:
                    st.error("❌ Model not loaded")
                else:
                    with st.spinner("🔄 Analyzing..."):
                        results = predict(model, predict_source)

                    st.markdown("---")
                    st.subheader("📊 Prediction Results")

                    # Top prediction
                    top_result = results[0]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🏆 Top Prediction", top_result["class"])
                    with col2:
                        st.metric("✅ Confidence", f"{top_result['confidence']:.2f}%")

                    # Health status
                    if "healthy" in top_result["class"].lower():
                        st.success("✅ Plant appears HEALTHY")
                    else:
                        st.error("⚠️ Disease DETECTED")

                    # Top 5 predictions
                    st.markdown("### 📈 Top 5 Predictions")
                    for i, result in enumerate(results, 1):
                        st.progress(result["confidence"] / 100)
                        st.write(f"**{i}. {result['class']}** - {result['confidence']:.2f}%")

    # Tab 2: File Structure
    with tab2:
        if image_dict:
            st.subheader("📂 File Structure")
            st.write(f"**Total Classes:** {len(image_dict)}")
            st.write(f"**Total Images:** {total_images}")
            st.markdown("---")

            for class_name in sorted(image_dict.keys()):
                st.write(f"**{class_name}** - {len(image_dict[class_name])} images")
        else:
            st.info("No sample images loaded")

    # Tab 3: Image Gallery
    with tab3:
        if image_dict:
            st.subheader("🖼️ Sample Image Gallery")
            display_file_structure(image_dict)
        else:
            st.info("No sample images loaded")

    # Tab 4: Info
    with tab4:
        st.subheader("ℹ️ Model Information")

        st.markdown("""
        ### 🌿 Plant Disease Detection CNN

        **Architecture:**
        - Input: 150x150x3 RGB images
        - Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
        - Flatten → Dense(32) → Dense(15, softmax)

        **Training:**
        - Dataset: PlantVillage (20,638 images)
        - 15 classes (Tomato, Potato, Pepper diseases + healthy)
        - Accuracy: ~98.7% validation

        **Classes:**
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Tomato (9)**")
            st.write("- Bacterial spot")
            st.write("- Early blight")
            st.write("- Late blight")
            st.write("- Leaf Mold")
            st.write("- Septoria leaf spot")
            st.write("- Spider mites")
            st.write("- Target Spot")
            st.write("- Yellow Leaf Curl Virus")
            st.write("- Mosaic virus")
            st.write("- Healthy")

        with col2:
            st.markdown("**Potato (3)**")
            st.write("- Early blight")
            st.write("- Late blight")
            st.write("- Healthy")

        with col3:
            st.markdown("**Pepper (2)**")
            st.write("- Bacterial spot")
            st.write("- Healthy")

if __name__ == "__main__":
    main()
