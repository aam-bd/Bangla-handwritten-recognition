"""
Word and character segmentation engine for handwritten Bangla script.
Handles stroke normalization, headline (matra) analysis, connected components,
and left-to-right character bounding box extraction.
"""

from typing import List, Dict, Any, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.dataset import preprocess_image


def prepare_canvas_image(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Standardize raw canvas drawing:
    Returns grayscale uint8 array where strokes are white (255) on black (0).
    """
    if isinstance(image, Image.Image):
        img_arr = np.array(image)
    else:
        img_arr = np.array(image, copy=True)

    if img_arr.ndim == 3:
        if img_arr.shape[2] == 4:  # RGBA from canvas
            # Check alpha channel
            alpha = img_arr[:, :, 3]
            rgb = img_arr[:, :, :3]
            # If canvas drew strokes with alpha > 0
            if np.max(alpha) > 0 and np.mean(alpha) < 250:
                # Strokes are where alpha > 30
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                # If RGB was black stroke, invert
                stroke_mask = (alpha > 30) & (gray < 200)
                if np.sum(stroke_mask) > 0:
                    binary = np.zeros(alpha.shape, dtype=np.uint8)
                    binary[stroke_mask] = 255
                    return binary
            # Otherwise standard composite
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        elif img_arr.shape[2] == 3:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_arr.squeeze(axis=2)
    else:
        gray = img_arr

    gray = gray.astype(np.uint8)

    # Invert if background is light (mean > 127)
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    return gray


def detect_matra_row(binary_image: np.ndarray) -> int:
    """
    Detect the index of the top headline (Matra) row using horizontal projection.
    Returns -1 if no prominent headline is detected.
    """
    h, w = binary_image.shape
    if h < 20 or w < 20:
        return -1

    # Find upper and lower bounds of foreground
    rows_with_ink = np.where(np.any(binary_image > 0, axis=1))[0]
    if len(rows_with_ink) < 10:
        return -1

    top_ink = rows_with_ink[0]
    bottom_ink = rows_with_ink[-1]
    ink_height = bottom_ink - top_ink

    # Focus on top 10% - 40% of character height
    search_start = top_ink + int(0.10 * ink_height)
    search_end = top_ink + int(0.40 * ink_height)
    if search_end <= search_start:
        return -1

    proj = np.sum(binary_image[search_start:search_end, :] > 0, axis=1)
    if len(proj) == 0:
        return -1

    max_idx = np.argmax(proj)
    max_val = proj[max_idx]

    # Headline usually spans a substantial portion of the word width
    if max_val > 0.35 * w:
        return search_start + max_idx
    return -1


def segment_word(
    image: Union[np.ndarray, Image.Image],
    min_area: int = 25,
    target_size: Tuple[int, int] = (64, 64)
) -> List[Dict[str, Any]]:
    """
    Segment a handwritten Bangla word image into ordered character bounding boxes.
    Returns a list of dicts:
    [
        {
            "bbox": (x, y, w, h),
            "image": np.ndarray (shape: (64, 64, 1), normalized [0, 1]),
            "raw_crop": np.ndarray (original grayscale crop)
        },
        ...
    ]
    Sorted horizontally from left to right.
    """
    gray = prepare_canvas_image(image)

    # Threshold
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    h, w = thresh.shape

    # Check if canvas is completely empty
    if np.sum(thresh > 0) < min_area:
        return []

    # Check for Matra (headline)
    matra_y = detect_matra_row(thresh)

    segmented_mask = thresh.copy()
    if matra_y > 0:
        # Temporarily suppress 2-3 pixels at the matra row to separate connected characters
        y1 = max(0, matra_y - 1)
        y2 = min(h, matra_y + 2)
        segmented_mask[y1:y2, :] = 0

    # Find connected components on separated mask
    contours, _ = cv2.findContours(segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch >= min_area and (cw >= 4 or ch >= 4):
            # If matra was suppressed, extend height upwards to recover headline
            if matra_y > 0 and y <= matra_y + 3 and y > 0:
                diff = y - max(0, matra_y - 2)
                y = max(0, matra_y - 2)
                ch = ch + diff
            raw_boxes.append((x, y, cw, ch))

    if not raw_boxes:
        # Fallback to bounding box of the entire image if contours failed
        ys, xs = np.where(thresh > 0)
        if len(xs) > 0:
            x, y, cw, ch = np.min(xs), np.min(ys), np.max(xs) - np.min(xs) + 1, np.max(ys) - np.min(ys) + 1
            raw_boxes.append((int(x), int(y), int(cw), int(ch)))
        else:
            return []

    # Sort boxes left-to-right
    raw_boxes.sort(key=lambda b: b[0])

    # Merge overlapping or vertically stacked fragments (e.g., upper/lower Kar diacritics)
    merged_boxes: List[Tuple[int, int, int, int]] = []
    for b in raw_boxes:
        if not merged_boxes:
            merged_boxes.append(b)
            continue

        prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
        curr_x, curr_y, curr_w, curr_h = b

        # Calculate horizontal overlap
        overlap_start = max(prev_x, curr_x)
        overlap_end = min(prev_x + prev_w, curr_x + curr_w)
        overlap = max(0, overlap_end - overlap_start)
        min_w = min(prev_w, curr_w)

        # If significant horizontal overlap (>40% of smaller box) or tiny sub-stroke
        if overlap > 0.40 * min_w or (curr_w < 12 and overlap > 0):
            new_x = min(prev_x, curr_x)
            new_y = min(prev_y, curr_y)
            new_w = max(prev_x + prev_w, curr_x + curr_w) - new_x
            new_h = max(prev_y + prev_h, curr_y + curr_h) - new_y
            merged_boxes[-1] = (new_x, new_y, new_w, new_h)
        else:
            merged_boxes.append(b)

    # Extract cropped images and normalize
    results = []
    for (x, y, cw, ch) in merged_boxes:
        # Extract from original clean stroke map
        pad_x = max(2, int(0.05 * cw))
        pad_y = max(2, int(0.05 * ch))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)

        crop = gray[y1:y2, x1:x2]
        processed_crop = preprocess_image(crop, target_size=target_size)

        results.append({
            "bbox": (x, y, cw, ch),
            "image": processed_crop,
            "raw_crop": crop
        })

    return results


def visualize_segments(
    image: Union[np.ndarray, Image.Image],
    segments: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Generate an accessible RGB visualization showing detected bounding boxes
    and character index badges overlaid on the drawing.
    """
    gray = prepare_canvas_image(image)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    colors = [
        (0, 200, 255),   # Golden Yellow
        (0, 255, 100),   # Neon Green
        (255, 100, 100), # Sky Blue
        (255, 150, 255), # Light Violet
        (100, 220, 255)  # Cyan
    ]

    for idx, seg in enumerate(segments):
        x, y, w, h = seg["bbox"]
        color = colors[idx % len(colors)]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        # Add index badge
        badge_text = f"C{idx + 1}"
        cv2.putText(vis, badge_text, (x + 2, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis
