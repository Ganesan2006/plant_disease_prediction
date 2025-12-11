import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom CSS for colorful design
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF6B6B 0%, #FFE66D 100%);
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 25px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .title {
        text-align: center;
        color: white;
        font-size: 48px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #FFE66D;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .real-news {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: slideIn 0.5s ease;
    }
    .fake-news {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: slideIn 0.5s ease;
    }
    .result-text {
        color: white;
        font-size: 36px;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .confidence {
        color: white;
        font-size: 18px;
        margin-top: 10px;
        opacity: 0.9;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .stTextArea textarea {
        border-radius: 15px;
        border: 3px solid #FFE66D;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_resources():
    try:
        model = load_model('fake_news_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Prediction function
def predict_news(text, model, tokenizer, max_length=500):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    return prediction

# Header
st.markdown('<p class="title">📰 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🔍 Verify the authenticity of news articles using AI</p>', unsafe_allow_html=True)

# Load model
model, tokenizer = load_resources()

if model is not None and tokenizer is not None:
    # Text input
    news_text = st.text_area(
        "Enter the news article or headline:",
        height=200,
        placeholder="Paste your news text here...",
        help="Enter the complete news article or headline you want to verify"
    )
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔎 Analyze News", use_container_width=True)
    
    # Prediction
    if predict_button:
        if news_text.strip():
            with st.spinner("🤖 AI is analyzing the news..."):
                prediction = predict_news(news_text, model, tokenizer)
                confidence = prediction if prediction > 0.5 else 1 - prediction
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if prediction > 0.5:
                    st.markdown(f"""
                        <div class="real-news">
                            <p class="result-text">✅ REAL NEWS</p>
                            <p class="confidence">Confidence: {confidence*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="fake-news">
                            <p class="result-text">❌ FAKE NEWS</p>
                            <p class="confidence">Confidence: {confidence*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: white; opacity: 0.7;'>
            <p>💡 Tip: For best results, provide complete sentences or full articles</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.error("❌ Failed to load the model. Please ensure 'fake_news_model.h5' and 'tokenizer.pkl' are in the same directory.")

