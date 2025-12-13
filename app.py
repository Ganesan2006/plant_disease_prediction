import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import zipfile
import os
import shutil
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ⭐ CRITICAL: Class order MUST match training order (alphabetically sorted)
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

NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "plant_disease_model.keras"
ZIP_FILE = "plant_images.zip"
SAMPLE_DIR = "sample_images"

# --------------------------------------------------
# Session state for sample images
# --------------------------------------------------
if "samples_loaded" not in st.session_state:
    st.session_state.samples_loaded = False
if "sample_dict" not in st.session_state:
    st.session_state.sample_dict = {}
if "load_attempted" not in st.session_state:
    st.session_state.load_attempted = False

# --------------------------------------------------
# Helper: Flatten nested folder structure
# --------------------------------------------------
def flatten_extracted_folder():
    """
    ZIP may create nested structure: some-folder/sample_images/class/image.jpg
    We need: sample_images/class/image.jpg
    This function flattens the structure if needed.
    """
    base_dir = SAMPLE_DIR

    if not os.path.exists(base_dir):
        return

    items = os.listdir(base_dir)

    # If only one folder inside and it's not a class name
    if len(items) == 1 and items[0] not in CLASS_NAMES:
        nested_path = os.path.join(base_dir, items[0])
        if os.path.isdir(nested_path):
            nested_items = os.listdir(nested_path)

            # Check if this contains sample_images folder
            if "sample_images" in nested_items:
                # Move sample_images content up
                inner_sample_path = os.path.join(nested_path, "sample_images")
                temp_dir = os.path.join(base_dir, "temp_move")
                shutil.move(inner_sample_path, temp_dir)
                shutil.rmtree(base_dir)
                shutil.move(temp_dir, base_dir)

            # Otherwise, check if class folders exist directly
            elif any(item in nested_items for item in CLASS_NAMES):
                # Move all class folders up one level
                for item in nested_items:
                    if item in CLASS_NAMES:
                        src = os.path.join(nested_path, item)
                        dst = os.path.join(base_dir, item)
                        if not os.path.exists(dst):
                            shutil.move(src, dst)

                # Remove the now-empty nested folder
                try:
                    shutil.rmtree(nested_path)
                except:
                    pass

# --------------------------------------------------
# Helper: Scan for sample images
# --------------------------------------------------
def scan_sample_images():
    """
    Scan sample_images folder and organize by class.
    Returns dict: {class_name: [list of image paths]}
    """
    sample_dict = {}

    if not os.path.exists(SAMPLE_DIR):
        return sample_dict

    for class_name in CLASS_NAMES:
        class_path = Path(SAMPLE_DIR) / class_name
        if class_path.exists() and class_path.is_dir():
            # Find all image files
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images.extend(class_path.glob(ext))

            if images:
                sample_dict[class_name] = [str(img) for img in sorted(images)]

    return sample_dict

# --------------------------------------------------
# Helper: Extract ZIP from local repository
# --------------------------------------------------
def extract_local_zip(zip_path: str):
    """
    Extract plant_images.zip from local repository.
    Handles nested folder structure from ZIP exports.
    """
    try:
        if not os.path.exists(zip_path):
            return False, f"❌ ZIP file not found: {zip_path}"

        with st.spinner("📦 Extracting sample images from ZIP..."):
            # Clean existing folder
            if os.path.exists(SAMPLE_DIR):
                shutil.rmtree(SAMPLE_DIR)

            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(SAMPLE_DIR)

            # Flatten structure if needed
            flatten_extracted_folder()

            # Scan for images
            st.session_state.sample_dict = scan_sample_images()
            st.session_state.samples_loaded = True

            total_imgs = sum(len(v) for v in st.session_state.sample_dict.values())

            if total_imgs == 0:
                return False, "⚠️ ZIP extracted but no images found. Check folder structure."

            return True, f"✅ Successfully extracted! {len(st.session_state.sample_dict)} classes, {total_imgs} images found"

    except zipfile.BadZipFile:
        return False, "❌ Invalid or corrupted ZIP file"
    except PermissionError:
        return False, "❌ Permission denied accessing ZIP file"
    except Exception as e:
        return False, f"❌ Error extracting ZIP: {str(e)}"

# --------------------------------------------------
# Load sample images (auto-run once)
# --------------------------------------------------
def load_sample_images():
    """Auto-load sample images on first run"""
    if not st.session_state.load_attempted:
        st.session_state.load_attempted = True

        if os.path.exists(ZIP_FILE):
            success, message = extract_local_zip(ZIP_FILE)
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.warning(message)
                st.sidebar.info("💡 You can still upload your own images!")
        else:
            st.sidebar.warning(f"⚠️ ZIP file '{ZIP_FILE}' not found")
            st.sidebar.info("💡 Upload your own images to use the app!")

    return st.session_state.sample_dict

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# --------------------------------------------------
# Preprocess image
# --------------------------------------------------
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

# --------------------------------------------------
# Make prediction
# --------------------------------------------------
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

# --------------------------------------------------
# Format disease name
# --------------------------------------------------
def format_disease_name(class_name):
    """Format class name for better display"""
    formatted = class_name.replace('_', ' ').replace('  ', ' - ')
    return formatted

# --------------------------------------------------
# Display image gallery
# --------------------------------------------------
def display_image_gallery(image_dict):
    """Display sample image gallery"""
    st.markdown("### 🖼️ Sample Image Gallery")

    for class_name in sorted(image_dict.keys()):
        with st.expander(f"**{format_disease_name(class_name)}** ({len(image_dict[class_name])} images)"):
            cols = st.columns(4)
            for idx, img_path in enumerate(image_dict[class_name][:20]):  # Show first 20
                with cols[idx % 4]:
                    try:
                        img = Image.open(img_path)
                        st.image(img, caption=Path(img_path).name, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")

# --------------------------------------------------
# Main app
# --------------------------------------------------
def main():
    st.title("🌿 Plant Disease Detection")
    st.markdown("---")

    # Load sample images
    image_dict = load_sample_images()
    total_images = sum(len(v) for v in image_dict.values()) if image_dict else 0

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")

        if image_dict:
            st.success(f"✅ Classes: {len(image_dict)}")
            st.success(f"📷 Images: {total_images}")
        else:
            st.info("💡 Upload images to predict")

        st.markdown("---")
        st.markdown("**Model:** CNN (150×150)")
        st.markdown("**Classes:** 15 diseases")
        st.markdown("**Accuracy:** ~98.7%")

        # Debug info
        with st.expander("🔧 Debug Info"):
            st.write("**Files in directory:**")
            files = os.listdir(".")[:15]
            for f in files:
                st.text(f"  {f}")
            st.write(f"**ZIP exists:** {os.path.exists(ZIP_FILE)}")
            if os.path.exists(ZIP_FILE):
                size_mb = os.path.getsize(ZIP_FILE) / (1024 * 1024)
                st.write(f"**ZIP size:** {size_mb:.2f} MB")
            st.write(f"**Sample folder exists:** {os.path.exists(SAMPLE_DIR)}")
            if os.path.exists(SAMPLE_DIR):
                sample_contents = os.listdir(SAMPLE_DIR)[:10]
                st.write(f"**Sample folder contents:** {sample_contents}")

    # Tabs
    if image_dict:
        tabs = st.tabs(["🔍 Predict", "📊 Classes", "🖼️ Gallery", "ℹ️ Info"])
    else:
        tabs = st.tabs(["🔍 Predict", "ℹ️ Info"])

    # Tab 1: Prediction
    with tabs[0]:
        st.header("🔬 Disease Prediction")

        col1, col2 = st.columns([1, 1])

        # Column 1: Upload
        with col1:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

            predict_img = None
            img_source = None

            if uploaded_file:
                predict_img = Image.open(uploaded_file)
                img_source = "uploaded"
                st.image(predict_img, caption="Uploaded Image", use_container_width=True)

        # Column 2: Select sample
        with col2:
            if image_dict:
                st.subheader("📁 Or Select Sample")
                selected_class = st.selectbox("Disease Class", options=sorted(image_dict.keys()))

                if selected_class and image_dict[selected_class]:
                    selected_image_path = st.selectbox(
                        "Image", 
                        options=image_dict[selected_class],
                        format_func=lambda x: Path(x).name
                    )

                    if selected_image_path and not uploaded_file:
                        predict_img = Image.open(selected_image_path)
                        img_source = "sample"
                        st.image(predict_img, caption="Sample Image", use_container_width=True)
            else:
                st.info("💡 No samples. Upload your own image!")

        st.markdown("---")

        # Predict button
        if st.button("🔬 Analyze Leaf", type="primary", use_container_width=True):
            if predict_img is None:
                st.warning("⚠️ Please upload or select an image first")
            else:
                model = load_model()
                if model is None:
                    st.error("❌ Model not loaded")
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

                    # Confidence bar
                    st.progress(top["confidence"] / 100, text=f"Confidence: {top['confidence']:.1f}%")

                    # Health message
                    if "healthy" in top["class"].lower():
                        st.success("✅ **Plant appears HEALTHY!** No disease detected.")
                    else:
                        st.error(f"⚠️ **Disease Detected:** {format_disease_name(top['class'])}")
                        st.info("💡 Recommend: Consult agricultural expert for treatment.")

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

    # Tab: Classes (if samples loaded)
    if image_dict and len(tabs) > 2:
        with tabs[1]:
            st.subheader("📂 Dataset Classes")
            st.write(f"**Total Classes:** {len(image_dict)}")
            st.write(f"**Total Images:** {total_images}")
            st.markdown("---")

            for class_name in sorted(image_dict.keys()):
                st.write(f"📁 **{format_disease_name(class_name)}** - {len(image_dict[class_name])} images")

    # Tab: Gallery (if samples loaded)
    if image_dict and len(tabs) > 2:
        with tabs[2]:
            display_image_gallery(image_dict)

    # Tab: Info
    with tabs[-1]:
        st.subheader("ℹ️ Model Information")

        st.markdown("""
        ### 🌿 Plant Disease Detection CNN

        **Architecture:**
        - Input: 150×150 RGB images
        - Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
        - Flatten → Dense(32) → Dense(15, softmax)

        **Training:**
        - Dataset: PlantVillage (20,638 images)
        - Optimizer: Adam
        - Loss: Categorical Crossentropy
        - Validation Accuracy: 98.59%
        - Training: 10 epochs

        **Supported Classes (15):**
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🍅 Tomato (9)**")
            for d in ["Bacterial spot", "Early blight", "Late blight", "Leaf Mold", 
                      "Septoria leaf spot", "Spider mites", "Target Spot", 
                      "Yellow Leaf Curl Virus", "Mosaic virus", "Healthy"]:
                st.write(f"- {d}")

        with col2:
            st.markdown("**🥔 Potato (3)**")
            for d in ["Early blight", "Late blight", "Healthy"]:
                st.write(f"- {d}")

        with col3:
            st.markdown("**🌶️ Pepper Bell (2)**")
            for d in ["Bacterial spot", "Healthy"]:
                st.write(f"- {d}")

        st.markdown("---")
        st.markdown("""
        **Usage Tips:**
        - Use clear, well-lit leaf images
        - Focus on diseased/healthy leaf areas
        - Avoid blurry or dark images
        - Best results with close-up shots
        """)

if __name__ == "__main__":
    main()
