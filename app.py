import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import zipfile
import shutil
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image or use sample images to detect diseases using a deep learning CNN model.")

# --------------------------------------------------
# Class names – MUST match training order (15 classes)
# --------------------------------------------------
class_names = [
    "Tomato__Tomato_mosaic_virus",
    "Potato___Early_blight",
    "Tomato_healthy",
    "Tomato_Septoria_leaf_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Potato___healthy",
    "Tomato_Leaf_Mold",
    "Tomato__Target_Spot",
    "Tomato_Late_blight",
    "Tomato_Early_blight",
    "Potato___Late_blight",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Pepper__bell___healthy",
    "Tomato_Bacterial_spot",
    "Pepper__bell___Bacterial_spot",
]

NUM_CLASSES = len(class_names)
SAMPLE_DIR = "sample_images"
ZIP_FILE = "plant_images.zip"

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
# Helper: Extract ZIP from local repository
# --------------------------------------------------
def extract_local_zip(zip_path: str):
    """
    Extract plant_disease.zip from local repository.
    Handles nested folder structure from ZIP exports.
    """
    try:
        if not os.path.exists(zip_path):
            return False, f"❌ ZIP file not found: {zip_path}"

        st.info("📦 Extracting sample images from ZIP...")

        # Clean existing folder
        if os.path.exists(SAMPLE_DIR):
            shutil.rmtree(SAMPLE_DIR)

        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(SAMPLE_DIR)

        # Flatten structure if needed
        flatten_extracted_folder()

        st.session_state.samples_loaded = True
        st.session_state.sample_dict = scan_sample_images()

        total_imgs = sum(len(v) for v in st.session_state.sample_dict.values())
        return True, f"✅ Successfully extracted! {len(st.session_state.sample_dict)} classes, {total_imgs} images found"

    except zipfile.BadZipFile:
        return False, "❌ Invalid or corrupted ZIP file"
    except PermissionError:
        return False, "❌ Permission denied accessing ZIP file"
    except Exception as e:
        return False, f"❌ Error extracting ZIP: {str(e)}"

def flatten_extracted_folder():
    """
    ZIP may create nested structure: some-folder/sample_images/class/image.jpg
    We need: sample_images/class/image.jpg
    This function flattens the structure if needed.
    """
    base_dir = SAMPLE_DIR
    items = os.listdir(base_dir)

    # If only one folder inside and it's not a class name
    if len(items) == 1 and items[0] not in class_names:
        nested_path = os.path.join(base_dir, items[0])
        if os.path.isdir(nested_path):
            # Check if this contains sample_images
            nested_items = os.listdir(nested_path)
            if "sample_images" in nested_items:
                # Move sample_images content up
                inner_sample_path = os.path.join(nested_path, "sample_images")
                temp_dir = os.path.join(base_dir, "temp_move")
                shutil.move(inner_sample_path, temp_dir)
                shutil.rmtree(base_dir)
                shutil.move(temp_dir, base_dir)
            # Otherwise, move all class folders up
            elif any(item in nested_items for item in class_names):
                temp_items = []
                for item in nested_items:
                    if item in class_names:
                        src = os.path.join(nested_path, item)
                        dst = os.path.join(base_dir, item)
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
                            temp_items.append(item)
                if temp_items:
                    shutil.rmtree(nested_path)

def scan_sample_images():
    """
    Scan SAMPLE_DIR and collect image paths for each class.
    Returns: {class_name: [image_paths]}
    """
    sample_dict = {}

    if not os.path.exists(SAMPLE_DIR):
        return sample_dict

    for class_name in class_names:
        class_path = os.path.join(SAMPLE_DIR, class_name)

        if os.path.isdir(class_path):
            images = []
            for fname in os.listdir(class_path):
                fpath = os.path.join(class_path, fname)
                if os.path.isfile(fpath) and fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(fpath)

            if images:
                sample_dict[class_name] = sorted(images)

    return sample_dict

# --------------------------------------------------
# AUTO-LOAD ZIP ON APP STARTUP
# --------------------------------------------------
if not st.session_state.load_attempted:
    st.session_state.load_attempted = True

    # Try to load plant_disease.zip from local repo
    if os.path.exists(ZIP_FILE):
        with st.spinner("🔄 Loading sample images from local ZIP..."):
            success, message = extract_local_zip(ZIP_FILE)
            if success:
                st.success(message)
            else:
                st.warning(message)
    else:
        st.info(f"ℹ️ Place '{ZIP_FILE}' in the app directory to auto-load sample images.")

# Load samples if not already loaded
if not st.session_state.samples_loaded:
    st.session_state.sample_dict = scan_sample_images()
    if st.session_state.sample_dict:
        st.session_state.samples_loaded = True

# --------------------------------------------------
# Model loading
# --------------------------------------------------
@st.cache_resource
def load_model():
    """Load trained Keras model"""
    try:
        # Try .keras first (TensorFlow 2.13+), then .h5
        if os.path.exists("plant_disease_model.keras"):
            model = keras.models.load_model("plantdiseasemodel.keras")
        else:
            st.error("❌ Model file not found (plantdiseasemodel.keras or plant_disease_model.h5)")
            return None
        return model
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None

# --------------------------------------------------
# Image preprocessing & prediction
# --------------------------------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert image to model input: 150x150 RGB, normalized [0,1]"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_resized = img.resize((150, 150))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(model, img: Image.Image):
    """Get prediction probabilities for image"""
    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(preds[0][idx])
    return class_names[idx], conf, preds[0]

# --------------------------------------------------
# Sidebar – Status & Manual reload
# --------------------------------------------------
with st.sidebar:
    st.header("📊 Status")

    if st.session_state.sample_dict:
        total_images = sum(len(imgs) for imgs in st.session_state.sample_dict.values())
        st.metric("✅ Classes loaded", len(st.session_state.sample_dict))
        st.metric("📷 Total images", total_images)

        if st.button("🔄 Reload ZIP", use_container_width=True):
            st.session_state.samples_loaded = False
            st.session_state.sample_dict = {}
            st.session_state.load_attempted = False
            st.rerun()
    else:
        st.warning("⚠️ No sample images loaded")
        if st.button("🔄 Reload ZIP", use_container_width=True):
            st.session_state.load_attempted = False
            st.rerun()

    st.divider()

    st.header("ℹ️ About")
    st.write("**Plant Disease Detector**")
    st.write("- 15 disease classes")
    st.write("- Input: 150×150 RGB")
    st.write("- Model: Deep CNN")
    st.write("- Auto-loads: plant_disease.zip")

# --------------------------------------------------
# Main tabs
# --------------------------------------------------
tab_predict, tab_samples, tab_gallery, tab_info = st.tabs([
    "🔍 Predict", 
    "📊 File Structure", 
    "🖼️ Image Gallery", 
    "ℹ️ Info"
])

# ==================================================
# TAB 1: PREDICT
# ==================================================
with tab_predict:
    st.subheader("Predict Plant Disease")

    col_input, col_output = st.columns(2)

    # Input column
    with col_input:
        st.write("### Upload or Select Sample")

        # Upload
        uploaded = st.file_uploader("📤 Upload leaf image", type=["jpg", "jpeg", "png"])

        st.divider()

        # Or use sample
        st.write("**Or use sample image:**")
        if st.session_state.sample_dict:
            selected_class = st.selectbox(
                "Select disease class",
                ["(none)"] + list(st.session_state.sample_dict.keys()),
                format_func=lambda x: "(No selection)" if x == "(none)" else x.replace("_", " ")
            )

            sample_path = None
            if selected_class != "(none)":
                images = st.session_state.sample_dict[selected_class]
                if images:
                    selected_image = st.selectbox(
                        "Select image file",
                        images,
                        format_func=lambda x: os.path.basename(x),
                        key="sample_select"
                    )
                    sample_path = selected_image
        else:
            st.info("📦 Sample images not loaded yet. Checking for plant_disease.zip...")
            selected_class = None
            sample_path = None

    # Decide which image to use
    image_to_process = None
    image_source = None

    if uploaded is not None:
        image_to_process = Image.open(uploaded)
        image_source = "Uploaded"
    elif sample_path is not None:
        image_to_process = Image.open(sample_path)
        image_source = "Sample"

    # Output column
    with col_output:
        if image_to_process is not None:
            st.image(image_to_process, caption=f"Input Image ({image_source})", use_container_width=True)

            model = load_model()
            if model is not None:
                st.write("### Prediction Result")

                with st.spinner("🔬 Analyzing..."):
                    pred_class, confidence, all_probs = predict_image(model, image_to_process)

                # Main prediction
                st.metric("Predicted Class", pred_class.replace("_", " "))
                st.metric("Confidence", f"{confidence*100:.2f}%")
                st.progress(confidence)

                # Health status
                if "healthy" in pred_class.lower():
                    st.success("✅ Leaf appears HEALTHY")
                else:
                    st.warning("⚠️ DISEASE DETECTED")

                # Top 5 probabilities
                st.write("**Top 5 predictions:**")
                top5_idx = np.argsort(all_probs)[-5:][::-1]
                for rank, idx in enumerate(top5_idx, 1):
                    prob_pct = all_probs[idx] * 100
                    st.write(f"{rank}. {class_names[idx].replace('_', ' ')} — {prob_pct:.2f}%")
        else:
            st.info("👆 Upload an image or select a sample image")

# ==================================================
# TAB 2: FILE STRUCTURE
# ==================================================
with tab_samples:
    st.subheader("📁 Sample Images File Structure")

    if not st.session_state.sample_dict:
        st.warning("No sample images loaded yet.")
        st.code(f"Looking for: {ZIP_FILE}")
    else:
        # Build file tree
        file_structure = f"sample_images/  ({len(st.session_state.sample_dict)} classes)\n"

        total_images = 0
        for i, class_name in enumerate(class_names):
            if class_name in st.session_state.sample_dict:
                images = st.session_state.sample_dict[class_name]
                total_images += len(images)

                is_last = (i == len(class_names) - 1)
                prefix = "└── " if is_last else "├── "
                file_structure += f"{prefix}{class_name}/  ({len(images)} images)\n"

                # Show first 3 image names
                for j, img_path in enumerate(images[:3]):
                    fname = os.path.basename(img_path)
                    is_last_img = (j == len(images[:3]) - 1) and (len(images) <= 3)
                    subprefix = "    └── " if is_last_img else "    ├── "
                    file_structure += f"{subprefix}{fname}\n"

                if len(images) > 3:
                    file_structure += f"    └── ... and {len(images) - 3} more\n"

        st.code(file_structure, language="text")

        st.divider()

        # Summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Classes", len(st.session_state.sample_dict))
        col2.metric("Total Images", total_images)
        col3.metric("Avg Images/Class", round(total_images / len(st.session_state.sample_dict), 1) if st.session_state.sample_dict else 0)

# ==================================================
# TAB 3: IMAGE GALLERY
# ==================================================
with tab_gallery:
    st.subheader("🖼️ Sample Images Gallery by Disease Class")

    if not st.session_state.sample_dict:
        st.warning("No sample images loaded yet.")
        st.info("📦 Ensure plant_disease.zip is in your repository directory")
    else:
        # Show all classes
        for class_name in class_names:
            if class_name not in st.session_state.sample_dict:
                continue

            images = st.session_state.sample_dict[class_name]
            num_images = len(images)

            with st.expander(f"📌 {class_name.replace('_', ' ')} ({num_images} images)", expanded=False):
                # Grid of images
                cols = st.columns(4)

                for idx, img_path in enumerate(images):
                    col = cols[idx % 4]
                    with col:
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                            st.caption(os.path.basename(img_path))
                        except Exception as e:
                            st.error(f"Could not load: {os.path.basename(img_path)}")

# ==================================================
# TAB 4: INFO
# ==================================================
with tab_info:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Model Architecture")
        st.code("""
Input: 150×150×3 (RGB image)
    ↓
Conv2D(32 filters, 3×3, ReLU)
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(64 filters, 3×3, ReLU)
    ↓
MaxPooling2D(2×2)
    ↓
Flatten
    ↓
Dense(32, ReLU)
    ↓
Dense(15, Softmax)
    ↓
Output: 15 classes
        """)

        st.subheader("📊 Performance")
        st.write("- **Training Accuracy:** ~98.7%")
        st.write("- **Validation Accuracy:** ~98.6%")
        st.write("- **Total Parameters:** 2.67M")

    with col2:
        st.subheader("🌱 Disease Classes (15)")

        # Organize by plant
        plants = {
            "🍅 Tomato": [c for c in class_names if "Tomato" in c],
            "🥔 Potato": [c for c in class_names if "Potato" in c],
            "🫑 Pepper": [c for c in class_names if "Pepper" in c],
        }

        for plant, classes in plants.items():
            with st.expander(plant):
                for cls in classes:
                    disease = cls.split("___")[-1]
                    st.write(f"- {disease.replace('_', ' ')}")

    st.divider()

    st.subheader("💡 Usage Tips")
    st.write("""
    ✓ **Image quality matters:**
      - Use clear, well-lit photos
      - Focus on the leaf area
      - Avoid motion blur
      - Single leaf works best

    ✓ **For best results:**
      - Plain background preferred
      - Capture both healthy & diseased areas
      - Use natural lighting
      - Avoid shadows

    ✓ **Model limitations:**
      - Trained on 15 disease classes
      - Works best for tomato, potato, pepper
      - Confidence scores guide reliability
      - Always verify with expert opinion
    """)

    st.divider()

    st.subheader("📦 Repository Setup")
    st.write(f"""
    **Required file:** `{ZIP_FILE}`

    Your repository should have:
    ```
    your-repo/
    ├── app.py
    ├── requirements.txt
    ├── plant_disease_model.h5 (or .keras)
    └── plant_disease.zip (contains sample images)
    ```

    **ZIP structure inside:**
    ```
    plant_disease.zip
    └── sample_images/
        ├── Tomato__Tomato_mosaic_virus/
        ├── Potato___Early_blight/
        ├── ... (15 folders)
        └── Pepper__bell___Bacterial_spot/
    ```
    """)
