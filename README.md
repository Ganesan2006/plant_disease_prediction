# 🌿 Plant Disease Detection – Streamlit App

A deep learning web application to detect plant diseases from leaf images using a CNN trained on the PlantVillage dataset.  
The app supports **15 disease classes** across Tomato, Potato, and Pepper plants, with **~98% validation accuracy**.

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)

---

## 🎯 Features

✅ **Upload & Predict** – Upload leaf images for instant disease detection  
✅ **Sample Gallery** – Browse pre-loaded sample images from 15 classes  
✅ **Top-5 Predictions** – See confidence scores for top likely diseases  
✅ **Smart Status** – Clear "Healthy ✅" vs "Disease ⚠️" diagnosis  
✅ **Nested ZIP Support** – Automatically handles any ZIP folder structure  
✅ **Works Offline & Online** – Deploy locally or on Streamlit Cloud  
✅ **Debug Tools** – See system info and troubleshoot easily  

---

## 🧠 Model Architecture

| Property | Details |
|----------|---------|
| **Framework** | TensorFlow / Keras |
| **Input Size** | 150×150 RGB images |
| **Layers** | Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(32) → Dense(15) |
| **Loss Function** | Categorical Crossentropy |
| **Optimizer** | Adam |
| **Dataset** | PlantVillage (~20,638 images) |
| **Validation Accuracy** | **98.59%** |
| **Training Epochs** | 10 |

### 15 Supported Classes

**🍅 Tomato (10 classes)**
- Bacterial spot
- Early blight
- Late blight
- Leaf Mold
- Septoria leaf spot
- Spider mites (Two-spotted)
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic virus
- Healthy

**🥔 Potato (3 classes)**
- Early blight
- Late blight
- Healthy

**🌶️ Pepper Bell (2 classes)**
- Bacterial spot
- Healthy

---

## 📂 Repository Structure

```
your-repo/
├── app.py                      # Main Streamlit application
├── plant_disease_model.keras   # Trained Keras/TensorFlow model
├── plant_images.zip            # (Optional) Sample images for testing
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📦 ZIP File Structure (Optional)

If you include `plant_images.zip`, it can be organized in any of these formats – the app handles all automatically:

### Format 1: With sample_images folder
```
plant_images.zip/
└── sample_images/
    ├── Tomato_Early_blight/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Tomato_Late_blight/
    ├── Potato___Early_blight/
    ├── Pepper__bell___Bacterial_spot/
    └── ... (15 folders total)
```

### Format 2: Nested repository export
```
plant_images.zip/
└── repo-main/
    └── sample_images/
        ├── Tomato_Early_blight/
        └── ... (15 folders)
```

### Format 3: Direct class folders
```
plant_images.zip/
├── Tomato_Early_blight/
├── Tomato_Late_blight/
├── Potato___Early_blight/
└── ... (15 folders directly)
```

✅ **All formats are automatically detected and flattened by the app!**

---

## 🚀 Installation & Local Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone Repository
```bash
git clone https://github.com/Ganesan2006/plant_disease_prediction.git
cd plant_disease_prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the App
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 🌐 Deploy on Streamlit Cloud

### Step 1: Push to GitHub
Ensure your repository contains:
- `app.py`
- `plant_disease_model.keras`
- `requirements.txt`
- (Optional) `plant_images.zip`

```bash
git add .
git commit -m "Initial commit - Plant Disease Detection app"
git push origin main
```

### Step 2: Deploy
1. Visit: https://share.streamlit.io
2. Click **"New app"**
3. Select your GitHub repo
4. Choose **Branch:** `main`
5. Choose **Main file path:** `app.py`
6. Click **Deploy**

✅ Your app is now live! Share the URL with others.

---

## 💻 How to Use

### Upload Your Own Image
1. Open the app in your browser
2. Go to **🔍 Predict** tab
3. Click **"📤 Upload Image"** section
4. Choose a JPG, JPEG, or PNG leaf image
5. Click **"🔬 Analyze Leaf"** button
6. View results with confidence scores

### Use Sample Images (if available)
1. Go to **📁 Or Select Sample** section
2. Choose a disease class from the dropdown
3. Select an image from the class
4. Click **"🔬 Analyze Leaf"** button
5. Get instant prediction

### Explore Samples
- **📊 Classes** tab – See how many images per class
- **🖼️ Gallery** tab – Browse all sample images
- **ℹ️ Info** tab – Learn about the model

---

## 📋 Requirements

All dependencies are listed in `requirements.txt`:

```
streamlit>=1.28.0
tensorflow>=2.13.0
tensorflow-hub
pillow>=9.0.0
numpy>=1.24.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

### Model Path
Update the `MODEL_PATH` variable in `app.py` if your model has a different name:
```python
MODEL_PATH = "plant_disease_model.keras"
```

### Sample Images ZIP
If your ZIP file has a different name, update:
```python
ZIP_FILE = "plant_images.zip"
```

### Sample Directory
If you want to extract to a different folder:
```python
SAMPLE_DIR = "sample_images"
```

---

## 🐛 Troubleshooting

### ⚠️ "ZIP file not found"
**Solution:** 
- Check that `plant_images.zip` is in the repository root
- Or remove the ZIP file entirely – the app works fine without it!
- Users can always upload their own images

### ⚠️ "Model not loaded"
**Solution:**
- Verify `plant_disease_model.keras` exists in the repo root
- Check file permissions
- Ensure TensorFlow is properly installed: `pip install tensorflow --upgrade`

### ⚠️ "No sample images available"
**Solution:**
- This is normal if `plant_images.zip` is missing
- Users can still upload and get predictions
- Add `plant_images.zip` to enable sample browsing

### 🔧 Debug Mode
Open **🔧 Debug Info** in the sidebar to see:
- Files in deployment directory
- ZIP file size
- Sample folder contents
- Extraction status

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Validation Accuracy | 98.59% |
| Training Accuracy | 98.70% |
| Validation Loss | 0.0398 |
| Training Loss | 0.0404 |
| Total Parameters | 2,674,127 |
| Model Size | ~10.2 MB |

---

## 🎨 UI Features

- **Responsive Design** – Works on desktop, tablet, and mobile
- **Dark Mode Support** – Respects system theme preferences
- **Progress Bars** – Visual confidence indicators
- **Status Badges** – Quick health/disease status
- **Expandable Sections** – Organized information layout

---

## 📝 Important Notes

⚠️ **Educational Purpose**: This tool is for learning and experimental use.  
⚠️ **Not Medical Advice**: Professional agricultural experts should confirm real-world diagnoses.  
⚠️ **Image Quality**: Results depend on clear, well-lit leaf photos.  
⚠️ **Limitations**: The model may struggle with:
  - Blurry images
  - Dark or poor lighting
  - Multiple diseases on one leaf
  - Plant species not in training data

---

## 👨‍💻 Technologies Used

- **TensorFlow/Keras** – Deep learning framework
- **Streamlit** – Web app framework
- **NumPy** – Numerical computing
- **Pillow** – Image processing
- **Python 3.8+** – Programming language

---

## 📄 License

This project is open-source. Feel free to use it for educational and commercial purposes.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📧 Contact & Support

- 📧 Email: ganesant2007@gmail.com
- 🐛 Issues: Open an issue on GitHub
- 💬 Discussions: Use GitHub Discussions

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** – For the training data
- **TensorFlow/Keras Team** – For the deep learning framework
- **Streamlit** – For the amazing web framework
- **Community** – For contributions and feedback

---

**Made with ❤️ for agriculture and machine learning**

Last Updated: December 2024
