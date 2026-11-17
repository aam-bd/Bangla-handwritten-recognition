"""
Whole-word recognition pipeline combining character segmentation,
deep neural classification, and lexicon-based sequence reconstruction.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
from PIL import Image

# Enforce backend
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
import keras

from src.segmenter import segment_word, visualize_segments
from src.lexicon import BanglaLexicon


class WordRecognizer:
    """
    End-to-end Bangla handwritten word recognition system:
    Canvas/Image -> Character Segmentation -> CNN Inference -> Lexicon Word Assembly.
    """

    def __init__(
        self,
        model_path: str = "models/model.keras",
        labels_path: str = "labels.json",
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.labels_path = labels_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load label mapping
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
        else:
            self.labels = {str(i): f"C{i}" for i in range(84)}

        self.lexicon = BanglaLexicon()
        self.model = None

    def _ensure_model_loaded(self):
        """Lazily load the Keras 3 model checkpoint."""
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model checkpoint not found at '{self.model_path}'. "
                    "Please run 'python train.py' to generate the model first."
                )
            self.model = keras.models.load_model(self.model_path)

    def predict_word(
        self,
        image: Union[np.ndarray, Image.Image],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Predict the complete Bangla word from a hand-drawn or scanned word image.
        Returns a rich prediction dictionary.
        """
        # 1. Segment drawing into ordered character components
        segments = segment_word(image)
        if not segments:
            return {
                "predicted_word": "",
                "confidence": 0.0,
                "raw_joined": "",
                "is_lexicon_match": False,
                "segments": [],
                "visualization": None,
                "message": "No handwriting detected on canvas."
            }

        self._ensure_model_loaded()

        # 2. Batch inference on segmented character crops
        batch_crops = np.stack([s["image"] for s in segments], axis=0)  # (N, 64, 64, 1)

        # Forward pass
        probs = self.model.predict(batch_crops, verbose=0)  # (N, 84)

        segment_results = []
        raw_chars = []
        confidences = []

        for idx, seg in enumerate(segments):
            p = probs[idx]
            top_indices = np.argsort(p)[::-1][:top_k]

            char_candidates = []
            for rank, c_idx in enumerate(top_indices):
                char_str = self.labels.get(str(c_idx), str(c_idx))
                conf = float(p[c_idx])
                char_candidates.append({
                    "char": char_str,
                    "confidence": conf,
                    "class_id": int(c_idx)
                })

            top_char = char_candidates[0]["char"]
            top_conf = char_candidates[0]["confidence"]
            raw_chars.append(top_char)
            confidences.append(top_conf)

            segment_results.append({
                "index": idx + 1,
                "bbox": seg["bbox"],
                "top_char": top_char,
                "confidence": top_conf,
                "candidates": char_candidates
            })

        # 3. Assemble whole word
        raw_joined = "".join(raw_chars)
        avg_char_conf = float(np.mean(confidences)) if confidences else 0.0

        # 4. Lexicon verification / spell refinement
        closest_word, match_score = self.lexicon.find_closest_word(raw_joined)

        if match_score >= 0.70:
            final_word = closest_word
            is_lexicon_match = True
            final_confidence = 0.6 * avg_char_conf + 0.4 * match_score
        else:
            final_word = raw_joined
            is_lexicon_match = False
            final_confidence = avg_char_conf

        # 5. Generate visual annotated overlay
        vis = visualize_segments(image, segments)

        return {
            "predicted_word": final_word,
            "confidence": round(float(final_confidence), 4),
            "raw_joined": raw_joined,
            "is_lexicon_match": is_lexicon_match,
            "segments": segment_results,
            "visualization": vis,
            "message": "Prediction successful"
        }
