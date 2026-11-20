# Shared helpers to draw face boxes and name labels whose size scales with
# the image resolution, so they stay readable on large images and videos.

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
REFERENCE_SIZE = 800.0  # lower = bigger boxes/labels, higher = smaller
BASE_FONT_SCALE = 0.4
BASE_THICKNESS = 2


def label_style(image):
    """Return (font_scale, thickness) proportional to the image size."""
    scale = max(image.shape[:2]) / REFERENCE_SIZE
    font_scale = max(BASE_FONT_SCALE, BASE_FONT_SCALE * scale)
    thickness = max(BASE_THICKNESS, int(round(BASE_THICKNESS * scale)))
    return font_scale, thickness


def draw_face(image, box, name, color, font_scale, thickness):
    """Draw the face box and a filled name label above it (or inside it)."""
    top, right, bottom, left = box
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)

    # filled background behind the label for readability
    (text_w, text_h), baseline = cv2.getTextSize(name, FONT, font_scale, thickness)
    pad = thickness * 2
    label_top = top - text_h - baseline - 2 * pad
    if label_top < 0:  # not enough room above the box: put the label inside it
        label_top = top
    label_bottom = label_top + text_h + baseline + 2 * pad
    cv2.rectangle(
        image, (left, label_top), (left + text_w + 2 * pad, label_bottom), color, -1
    )
    cv2.putText(
        image,
        name,
        (left + pad, label_bottom - baseline - pad),
        FONT,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
