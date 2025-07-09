import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Adjusted classification thresholds for white rice on black
MIN_AREA = 100          # Smaller to pick up thin grains
BROKEN_AREA = 350       # Slightly lower so short fragments count
BROKEN_LENGTH = 80      # Higher to catch shorter broken
MIN_ASPECT_RATIO = 2.0  # Slightly more forgiving
def process_image(path):
    import cv2
    import numpy as np

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image from {path}")

    result_img = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold (rice is white on black)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    # Contour detection
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store grain measurements
    grain_data = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (width, height), angle = rect
        length = max(width, height)
        aspect_ratio = length / (min(width, height) + 1e-5)

        grain_data.append({
            "contour": cnt,
            "rect": rect,
            "area": area,
            "length": length,
            "aspect_ratio": aspect_ratio
        })

    if not grain_data:
        # Nothing found
        summary = "No grains detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(summary, font, 0.6, 2)
        x, y = 10, result_img.shape[0] - 10
        cv2.rectangle(result_img,
                      (x - 5, y - text_h - 5),
                      (x + text_w + 5, y + 5),
                      (0, 0, 0), -1)
        cv2.putText(result_img, summary, (x, y),
                    font, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        return result_img

    # Adaptive threshold for broken vs whole
    lengths = sorted([g["length"] for g in grain_data], reverse=True)

    if len(lengths) <= 3:
        # Few grains → fixed ratio cutoff
        reference_length = lengths[0]
        broken_length_threshold = 0.75 * reference_length
    else:
        # Many grains → median of top 20%
        top_count = max(3, len(lengths) // 5)
        top_lengths = lengths[:top_count]
        reference_length = np.median(top_lengths)
        broken_length_threshold = 0.65 * reference_length

    broken_count = 0
    whole_count = 0

    for grain in grain_data:
        rect = grain["rect"]
        length = grain["length"]

        is_broken = length < broken_length_threshold

        if is_broken:
            color = (0, 0, 255)  # Red for broken
            broken_count += 1
        else:
            color = (0, 255, 0)  # Green for whole
            whole_count += 1

        box = cv2.boxPoints(rect)
        box = box.astype(int)
        cv2.drawContours(result_img, [box], 0, color, 2)

    # Draw summary
    summary = f"Broken: {broken_count}, Whole: {whole_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(summary, font, 0.6, 2)
    x, y = 10, result_img.shape[0] - 10

    cv2.rectangle(result_img,
                  (x - 5, y - text_h - 5),
                  (x + text_w + 5, y + 5),
                  (0, 0, 0), -1)

    cv2.putText(result_img, summary, (x, y),
                font, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    return result_img



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_img = process_image(filepath)
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
        cv2.imwrite(result_path, result_img)

        return render_template('index.html', uploaded_image=result_path)
    return render_template('index.html')

