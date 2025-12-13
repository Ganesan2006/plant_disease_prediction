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
EXTRACT_DIR = "sample_images"

# Auto-load sample images from ZIP
@st.cache_resource
def load_sample_images():
    """Extract sample images from ZIP file"""
    if not os.path.exists(ZIP_PATH):
        st.sidebar.warning(f"⚠️ ZIP file '{ZIP_PATH}' not found")
        st.sidebar.info("💡 Upload your own images to use the app!")
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
            # Try both with and without "sample_images" subfolder
            paths_to_try = [
                Path(EXTRACT_DIR) / "sample_images" / class_name,
                Path(EXTRACT_DIR) / class_name
            ]

            for class_path in paths_to_try:
                if class_path.exists():
                    images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.jpeg"))
                    if images:
                        image_dict[class_name] = [str(img) for img in images]
                        total_images += len(images)
                        break

        if image_dict:
            st.sidebar.success(f"✅ Loaded {len(image_dict)} classes, {total_images} images")

        return image_dict, total_images

    except Exception as e:
        st.sidebar.error(f"❌ Error extracting ZIP: {str(e)}")
        return None, None

# Load model
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Preprocess image
def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to 150x150 (training size)
    img = image.resize((150, 150))
    # Convert to array
    img_array = np.array(img)
    # Ensure 3 channels (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
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

# Format disease name for display
def format_disease_name(class_name):
    """Format class name for better display"""
    # Replace underscores with spaces
    formatted = class_name.replace('_', ' ').replace('  ', ' - ')
    return formatted

# Main app
def main():
    st.title("🌿 Plant Disease Detection")
    st.markdown("---")

    # Load sample images
    with st.spinner("🔄 Loading sample images..."):
        image_dict, total_images = load_sample_images()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")

        if image_dict:
            st.success(f"✅ Classes: {len(image_dict)}")
            st.success(f"📷 Images: {total_images}")
        else:
            st.info("💡 Upload images to predict")

        st.markdown("---")
        st.markdown("**Model:** CNN (150x150)")
        st.markdown("**Classes:** 15 diseases")
        st.markdown("**Accuracy:** ~98.7%")

        # Debug info (optional - remove in production)
        with st.expander("🔧 Debug Info"):
            st.write("Files in directory:", os.listdir(".")[:10])
            st.write(f"ZIP exists: {os.path.exists(ZIP_PATH)}")
            if os.path.exists(ZIP_PATH):
                st.write(f"ZIP size: {os.path.getsize(ZIP_PATH)/(1024*1024):.2f} MB")

    # Tabs
    if image_dict:
        tabs = st.tabs(["🔍 Predict", "📊 File Structure", "🖼️ Gallery", "ℹ️ Info"])
    else:
        tabs = st.tabs(["🔍 Predict", "ℹ️ Info"])

    # Tab 1: Prediction
    with tabs[0]:
        st.header("🔬 Disease Prediction")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image_to_predict = Image.open(uploaded_file)
                st.image(image_to_predict, caption="Uploaded Image", use_container_width=True)

        with col2:
            if image_dict:
                st.subheader("📁 Or Select Sample")
                selected_class = st.selectbox("Disease Class", options=sorted(image_dict.keys()))
                if selected_class:
                    selected_image_path = st.selectbox("Image", options=image_dict[selected_class])
                    if selected_image_path:
                        image_to_predict = Image.open(selected_image_path)
                        st.image(image_to_predict, caption="Sample Image", use_container_width=True)
            else:
                st.info("💡 No sample images available. Upload your own!")
                image_to_predict = None
                if not uploaded_file:
                    image_to_predict = None

        st.markdown("---")

        # Predict button
        if st.button("🔬 Analyze Leaf", type="primary", use_container_width=True):
            # Determine which image to predict
            if uploaded_file:
                predict_img = Image.open(uploaded_file)
            elif image_dict and 'selected_image_path' in locals():
                predict_img = Image.open(selected_image_path)
            else:
                st.warning("⚠️ Please upload or select an image first")
                predict_img = None

            if predict_img:
                model = load_model()
                if model is None:
                    st.error("❌ Model not loaded. Check MODEL_PATH.")
                else:
                    with st.spinner("🔄 Analyzing leaf..."):
                        results = predict(model, predict_img)

                    st.markdown("---")
                    st.subheader("📊 Analysis Results")

                    # Top prediction
                    top = results[0]
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("🏆 Prediction", format_disease_name(top["class"]))
                    with col2:
                        st.metric("📈 Confidence", f"{top['confidence']:.1f}%")
                    with col3:
                        if "healthy" in top["class"].lower():
                            st.metric("🏥 Status", "Healthy ✅")
                        else:
                            st.metric("🏥 Status", "Disease ⚠️")

                    # Visual confidence bar
                    st.progress(top["confidence"] / 100, text=f"Confidence: {top['confidence']:.1f}%")

                    # Health message
                    if "healthy" in top["class"].lower():
                        st.success("✅ **Plant appears HEALTHY!** No disease detected.")
                    else:
                        st.error(f"⚠️ **Disease Detected:** {format_disease_name(top['class'])}")
                        st.info("💡 Recommend: Consult agricultural expert for treatment options.")

                    # Top 5 predictions
                    st.markdown("---")
                    st.markdown("### 📊 Top 5 Predictions")

                    for i, result in enumerate(results, 1):
                        conf = result['confidence']
                        disease = format_disease_name(result['class'])

                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"**{i}. {disease}**")
                            st.progress(conf / 100)
                        with col_b:
                            st.write(f"**{conf:.1f}%**")

    # Tab: File Structure (if samples loaded)
    if image_dict and len(tabs) > 2:
        with tabs[1]:
            st.subheader("📂 Sample Dataset Structure")
            st.write(f"**Total Classes:** {len(image_dict)}")
            st.write(f"**Total Images:** {total_images}")
            st.markdown("---")

            for class_name in sorted(image_dict.keys()):
                st.write(f"📁 **{format_disease_name(class_name)}** - {len(image_dict[class_name])} images")

    # Tab: Gallery (if samples loaded)
    if image_dict and len(tabs) > 2:
        with tabs[2]:
            st.subheader("🖼️ Sample Image Gallery")
            display_file_structure(image_dict)

    # Tab: Info
    with tabs[-1]:
        st.subheader("ℹ️ Model Information")

        st.markdown("""
        ### 🌿 Plant Disease Detection System

        **Model Architecture:**
        - Input: 150×150 RGB images
        - Conv2D(32, 3×3) → MaxPool(2×2)
        - Conv2D(64, 3×3) → MaxPool(2×2)
        - Flatten → Dense(32) → Dense(15, softmax)

        **Training Details:**
        - Dataset: PlantVillage (20,638 images)
        - Optimizer: Adam
        - Loss: Categorical Crossentropy
        - Final Accuracy: 98.59% (validation)
        - Training Time: ~10 epochs

        **Supported Classes (15):**
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🍅 Tomato (9)**")
            tomato_diseases = [
                "Bacterial spot",
                "Early blight",
                "Late blight",
                "Leaf Mold",
                "Septoria leaf spot",
                "Spider mites",
                "Target Spot",
                "Yellow Leaf Curl Virus",
                "Mosaic virus",
                "Healthy"
            ]
            for d in tomato_diseases:
                st.write(f"- {d}")

        with col2:
            st.markdown("**🥔 Potato (3)**")
            potato_diseases = [
                "Early blight",
                "Late blight",
                "Healthy"
            ]
            for d in potato_diseases:
                st.write(f"- {d}")

        with col3:
            st.markdown("**🌶️ Pepper Bell (2)**")
            pepper_diseases = [
                "Bacterial spot",
                "Healthy"
            ]
            for d in pepper_diseases:
                st.write(f"- {d}")

        st.markdown("---")
        st.markdown("""
        **Usage Tips:**
        - Use clear, well-lit leaf images
        - Focus on diseased/healthy areas
        - Avoid blurry or dark images
        - Best results with close-up shots
        """)

if __name__ == "__main__":
    main()
