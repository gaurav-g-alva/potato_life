import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Potato Leaf Disease Detector",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ──────────────────────────────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH  = 224
CLASS_LABELS = ["Early Blight", "Late Blight", "Healthy"]   # alphabetical order (Keras default)

DISEASE_INFO = {
    "Early Blight": {
        "icon": "⚠️",
        "color": "#FF8C00",
        "description": (
            "Early Blight is caused by the fungus *Alternaria solani*. "
            "It appears as dark brown to black lesions with concentric rings "
            "forming a 'target-board' pattern on older leaves."
        ),
        "treatment": [
            "Remove and destroy infected leaves immediately.",
            "Apply fungicides such as chlorothalonil or mancozeb.",
            "Ensure proper plant spacing for airflow.",
            "Avoid overhead irrigation; water at the base.",
            "Rotate crops — avoid planting potatoes in the same spot each year.",
        ],
        "severity": "Moderate",
    },
    "Late Blight": {
        "icon": "🚨",
        "color": "#DC143C",
        "description": (
            "Late Blight is caused by *Phytophthora infestans*, the same pathogen "
            "responsible for the Irish Potato Famine. It spreads rapidly in cool, "
            "moist conditions and can destroy an entire crop within days."
        ),
        "treatment": [
            "Apply systemic fungicides (e.g., metalaxyl) at first sign.",
            "Remove and bag infected plants — do not compost.",
            "Avoid working in fields when foliage is wet.",
            "Use certified disease-free seed potatoes.",
            "Monitor weather — spray preventively during high-risk periods.",
        ],
        "severity": "Severe",
    },
    "Healthy": {
        "icon": "✅",
        "color": "#228B22",
        "description": (
            "The leaf appears healthy with no signs of disease. "
            "Continue good agricultural practices to maintain plant health."
        ),
        "treatment": [
            "Maintain regular watering and fertilisation schedule.",
            "Keep monitoring leaves weekly for early signs of disease.",
            "Ensure good soil drainage.",
            "Use balanced NPK fertilisers for robust growth.",
        ],
        "severity": "None",
    },
}

SEVERITY_BADGE = {
    "None":     ("🟢", "green"),
    "Moderate": ("🟡", "orange"),
    "Severe":   ("🔴", "red"),
}

# ─── Model Loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "potato_model.h5"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

# ─── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
   
    st.markdown("## 🥔 About This App")
    st.markdown(
        """
        This tool uses a **Convolutional Neural Network (CNN)** trained on the
        PlantVillage dataset to classify potato leaf images into one of three categories:

        | Class | Risk |
        |---|---|
        | 🟢 Healthy | None |
        | 🟡 Early Blight | Moderate |
        | 🔴 Late Blight | Severe |

        ---
        **Model Architecture**
        - 4 × Conv2D + MaxPooling blocks
        - Fully-connected Dense layers
        - Softmax output (3 classes)
        - Input size: 224 × 224 px

        ---
        **How to use**
        1. Upload a clear photo of a potato leaf.
        2. Click **Analyse Leaf**.
        3. View the prediction and recommended treatment.

        ---
        *Built with TensorFlow & Streamlit*
        """
    )

# ─── Main Content ───────────────────────────────────────────────────────────────
st.title("🥔 Potato Leaf Disease Detection")
st.markdown(
    "Upload a **potato leaf image** and the AI model will detect whether it is "
    "**Healthy**, affected by **Early Blight**, or affected by **Late Blight**."
)
st.divider()

# Load model
with st.spinner("Loading model…"):
    model = load_model()

if model is None:
    st.error(
        "⚠️  **Model file not found.**  "
        "Please place `potato_disease_model.h5` in the same directory as `app.py` and restart the app.",
        icon="🚫",
    )
    st.info(
        "**To save your trained model**, add this line at the end of your notebook:\n"
        "```python\ncnn_model.save('potato_disease_model.h5')\n```",
        icon="💡",
    )
    st.stop()

st.success("✅ Model loaded successfully!", icon="🧠")
st.divider()

# Upload section
col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("📤 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a potato leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        help="For best results, upload a clear close-up photo of a single leaf.",
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        analyse_btn = st.button("🔍 Analyse Leaf", type="primary", use_container_width=True)
    else:
        st.info("👆 Upload an image to get started.", icon="📁")
        analyse_btn = False

with col_preview:
    if uploaded_file:
        st.subheader("🖼️ Uploaded Image")
        st.image(img, use_container_width=False, caption=uploaded_file.name)

# ─── Prediction ─────────────────────────────────────────────────────────────────
if uploaded_file and analyse_btn:
    st.divider()
    st.subheader("🧪 Analysis Results")

    with st.spinner("Analysing leaf…"):
        time.sleep(0.4)  # small UX pause
        img_array   = preprocess_image(img)
        predictions = model.predict(img_array, verbose=0)[0]

    pred_idx   = int(np.argmax(predictions))
    pred_class = CLASS_LABELS[pred_idx]
    confidence = float(predictions[pred_idx]) * 100
    info       = DISEASE_INFO[pred_class]
    sev_icon, sev_color = SEVERITY_BADGE[info["severity"]]

    # ── Result banner ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {info['color']}22, {info['color']}11);
            border-left: 6px solid {info['color']};
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
        ">
            <h2 style="margin:0; color:{info['color']};">
                {info['icon']} {pred_class}
            </h2>
            <p style="margin:4px 0 0; font-size:1.1rem;">
                Confidence: <strong>{confidence:.1f}%</strong> &nbsp;|&nbsp;
                Severity: <strong style="color:{sev_color};">{sev_icon} {info['severity']}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Two-column detail ──────────────────────────────────────────────────────
    col_info, col_chart = st.columns([1.2, 1], gap="large")

    with col_info:
        st.markdown("#### 📋 Description")
        st.markdown(info["description"])

        st.markdown("#### 💊 Recommended Actions")
        for tip in info["treatment"]:
            st.markdown(f"- {tip}")

    with col_chart:
        st.markdown("#### 📊 Confidence Scores")
        import pandas as pd

        chart_df = pd.DataFrame(
            {
                "Disease Class": CLASS_LABELS,
                "Confidence (%)": [round(float(p) * 100, 2) for p in predictions],
            }
        ).set_index("Disease Class")

        st.bar_chart(chart_df, height=260, color="#4CAF50")

        st.markdown("#### 🔢 Raw Probabilities")
        for label, prob in zip(CLASS_LABELS, predictions):
            bar_pct = int(float(prob) * 100)
            st.markdown(
                f"**{label}** &nbsp; `{float(prob)*100:.2f}%`"
            )
            st.progress(bar_pct)

   
  