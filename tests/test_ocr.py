"""
Unit tests for Bangla OCR system components:
- BanglaLexicon consonant skeleton mapping & spell refinement
- Word and character segmentation
- Model building and inference output verification
"""

import pytest
import numpy as np
import cv2

from src.lexicon import BanglaLexicon, get_consonant_skeleton
from src.segmenter import segment_word
from src.model import build_model


def test_consonant_skeleton_extraction():
    assert get_consonant_skeleton("মানুষ") == "মনষ"
    assert get_consonant_skeleton("জল") == "জল"
    assert get_consonant_skeleton("বাংলাদেশ") == "বংলদশ"
    assert get_consonant_skeleton("বই") == "বই"
    assert get_consonant_skeleton("বিজ্ঞান") == "বজঞন"


def test_lexicon_consonant_skeleton_matching():
    lexicon = BanglaLexicon()

    # Exact word match
    word, score = lexicon.find_closest_word("জল")
    assert word == "জল"
    assert score == 1.0

    # Consonant skeleton match for 'মানুষ' (without kars)
    word, score = lexicon.find_closest_word("মনষ")
    assert word == "মানুষ"
    assert score >= 0.90

    # Skeleton match when akar is detected as '১'
    word, score = lexicon.find_closest_word("ম১নষ")
    assert word == "মানুষ"
    assert score >= 0.85

    # Full word 'মানুষ'
    word, score = lexicon.find_closest_word("মানুষ")
    assert word == "মানুষ"
    assert score == 1.0


def test_model_architecture():
    model = build_model(input_shape=(64, 64, 1), num_classes=84)
    assert model is not None
    # Test forward pass with random batch
    dummy_input = np.random.rand(2, 64, 64, 1).astype(np.float32)
    output = model(dummy_input, training=False)
    # Check output shape
    assert output.shape == (2, 84)
    # Check probabilities sum to ~1.0
    row_sums = output.sum(dim=-1).detach().cpu().numpy()
    np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-5)


def test_segmenter_empty_and_canvas():
    # Empty canvas
    empty_canvas = np.zeros((100, 300), dtype=np.uint8)
    assert segment_word(empty_canvas) == []

    # Canvas with 2 simulated character blocks
    canvas = np.zeros((120, 300), dtype=np.uint8)
    cv2.circle(canvas, (80, 60), 25, 255, -1)
    cv2.circle(canvas, (200, 60), 25, 255, -1)
    segments = segment_word(canvas)
    assert len(segments) == 2
    # Ensure left-to-right order
    assert segments[0]["bbox"][0] < segments[1]["bbox"][0]
    assert segments[0]["image"].shape == (64, 64, 1)
