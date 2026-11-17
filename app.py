"""
Streamlit Web Application: Bangla Handwritten Word Recognition System.
Designed for high accessibility, reliability, and whole-word prediction.
Assignment 04 - SICIP, BRAC University.
"""

import os
import sys
from pathlib import Path

# Enforce backend
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import cv2
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Project imports
from src.recognizer import WordRecognizer
from src.dataset import preprocess_image


# Page Configuration
st.set_page_config(
    page_title="Bangla Handwritten Word Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Accessible CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans Bengali', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #5f6368;
        margin-bottom: 1.2rem;
    }
    .banner-card {
        background-color: #f8f9fa;
        border: 1px solid #dadce0;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 1.5rem;
    }
    .predicted-word-box {
        background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%);
        color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(26, 115, 232, 0.25);
        margin: 12px 0 20px 0;
    }
    .predicted-word-text {
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 0;
        line-height: 1.2;
    }
    .confidence-badge {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .char-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .char-display {
        font-size: 2.2rem;
        font-weight: 700;
        color: #202124;
    }
    .badge-success {
        color: #137333;
        background-color: #ceead6;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_recognizer() -> WordRecognizer:
    """Load and cache the WordRecognizer model instance."""
    return WordRecognizer(
        model_path="models/model.keras",
        labels_path="labels.json"
    )


def create_synthetic_word_sample(text: str) -> np.ndarray:
    """Generate a clean synthetic word canvas image for immediate demo testing."""
    canvas = np.zeros((150, 450), dtype=np.uint8)
    # Draw simple character blocks depending on word
    if text == "জল":
        # Block 1 (left): জ
        cv2.putText(canvas, "J", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 6)
        # Block 2 (right): ল
        cv2.putText(canvas, "L", (260, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 6)
    elif text == "মানুষ":
        cv2.putText(canvas, "M", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 5)
        cv2.putText(canvas, "A", (160, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 5)
        cv2.putText(canvas, "N", (260, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 5)
        cv2.putText(canvas, "S", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 5)
    else:
        cv2.putText(canvas, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 5)
    return canvas


def main():
    # Sidebar: Information & Accessibility Controls
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/e/eb/BRAC_University_logo.svg/300px-BRAC_University_logo.svg.png", width=140)
        st.markdown("### SICIP — Assignment 04")
        st.markdown("**Bangla Handwritten Word Recognition**")
        st.markdown("---")
        st.markdown("**Student Details:**")
        st.markdown("- **Name:** Md. Abdullah Al Mamun")
        st.markdown("- **ID:** `1000128810`")
        st.markdown("- **Institution:** BRAC University")
        st.markdown("---")

        st.markdown("### 🎨 Canvas Settings")
        stroke_width = st.slider("Pen Stroke Width", min_value=6, max_value=24, value=14, step=2)
        stroke_color = st.color_picker("Stroke Color", "#FFFFFF")
        bg_color = st.color_picker("Background Color", "#000000")

        st.markdown("---")
        st.markdown("### ℹ️ Target Words")
        st.info("Assignment Requirement:\nDrawing **মানুষ** must predict **মানুষ**.\nDrawing **জল** must predict **জল**.")

    # Header
    st.markdown('<div class="main-title">✍️ Bangla Handwritten Word Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">End-to-End Character Segmentation, CNN Inference & Lexicon Word Reconstruction</div>', unsafe_allow_html=True)

    # Main Interaction Tabs
    tab_draw, tab_upload, tab_presets = st.tabs([
        "✏️ Draw Word (Canvas)",
        "📁 Upload Image (Accessible Alternative)",
        "⭐ Preset Word Examples"
    ])

    input_image = None

    with tab_draw:
        st.markdown("**Draw a complete Bangla word horizontally on the canvas below:**")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=200,
            width=650,
            drawing_mode="freedraw",
            key="bangla_word_canvas",
        )
        if canvas_result is not None and canvas_result.image_data is not None:
            # Check if user actually drew something
            img_data = canvas_result.image_data
            if np.max(img_data[:, :, :3]) > 0 or np.max(img_data[:, :, 3]) > 0:
                input_image = img_data

    with tab_upload:
        st.markdown("**Upload a scanned or pre-drawn image containing a handwritten Bangla word:**")
        uploaded_file = st.file_uploader(
            "Choose an image file (PNG, JPG, BMP)",
            type=["png", "jpg", "jpeg", "bmp"],
            help="High-contrast black-and-white or color images supported."
        )
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            st.image(pil_img, caption="Uploaded Image", width=350)
            input_image = np.array(pil_img)

    with tab_presets:
        st.markdown("**Select a target assignment word to run instant end-to-end recognition:**")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("Test Target: 'মানুষ'", use_container_width=True):
                st.session_state["preset_word"] = "মানুষ"
        with col_p2:
            if st.button("Test Target: 'জল'", use_container_width=True):
                st.session_state["preset_word"] = "জল"
        with col_p3:
            if st.button("Test Target: 'বই'", use_container_width=True):
                st.session_state["preset_word"] = "বই"

        if "preset_word" in st.session_state:
            target_preset = st.session_state["preset_word"]
            st.success(f"Selected demo target: **{target_preset}**")
            input_image = create_synthetic_word_sample(target_preset)

    st.markdown("---")

    # Recognition Execution Block
    if input_image is not None:
        col_out, col_seg = st.columns([1.2, 1.0])

        try:
            recognizer = load_recognizer()
            result = recognizer.predict_word(input_image, top_k=3)

            with col_out:
                st.markdown("### 🎯 Predicted Whole Word")

                if result["predicted_word"]:
                    pred_word = result["predicted_word"]
                    conf = result["confidence"]
                    is_lex = result["is_lexicon_match"]

                    st.markdown(f"""
                    <div class="predicted-word-box">
                        <div class="predicted-word-text">{pred_word}</div>
                        <div class="confidence-badge">Confidence: {conf*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Status metrics
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        if is_lex:
                            st.markdown('<span class="badge-success">✓ Verified in Bangla Lexicon</span>', unsafe_allow_html=True)
                        else:
                            st.caption("ℹ️ Raw phoneme sequence")
                    with m_col2:
                        st.caption(f"Raw characters: `{result['raw_joined']}`")

                else:
                    st.warning("No clear handwritten strokes detected. Please draw a word on the canvas.")

            with col_seg:
                st.markdown("### 🔍 Character Segmentation")
                if result["visualization"] is not None:
                    st.image(result["visualization"], caption="Detected Character Bounding Boxes", use_container_width=True)

            # Detailed per-character breakdown
            if result["segments"]:
                st.markdown("### 🧩 Segmented Character Analysis")
                cols = st.columns(min(len(result["segments"]), 6))
                for i, seg in enumerate(result["segments"][:6]):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="char-card">
                            <div style="font-size: 0.85rem; color: #5f6368;">Character {seg['index']}</div>
                            <div class="char-display">{seg['top_char']}</div>
                            <div style="font-size: 0.85rem; font-weight: 600; color: #1a73e8;">{seg['confidence']*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        # Top-3 alternatives
                        alt_text = ", ".join([f"{c['char']} ({c['confidence']*100:.0f}%)" for c in seg["candidates"][1:3]])
                        if alt_text:
                            st.caption(f"Alt: {alt_text}")

        except Exception as e:
            st.error(f"Error during recognition: {e}")
            st.info("Ensure that 'models/model.keras' and 'labels.json' exist.")
    else:
        st.info("👆 Draw a word in the canvas above, upload an image, or pick a preset to see the prediction.")


if __name__ == "__main__":
    main()
