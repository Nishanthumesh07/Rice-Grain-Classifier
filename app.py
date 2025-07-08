import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Classification thresholds
MIN_AREA = 300
BROKEN_AREA = 500
BROKEN_LENGTH = 60
MIN_ASPECT_RATIO = 2.2

def process_image(path):
    img = cv2.imread(path)
    result_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold with Otsu
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphology to remove noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    broken_count = 0
    whole_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (width, height), angle = rect
        length = max(width, height)
        aspect_ratio = length / (min(width, height) + 1e-5)

        is_broken = (area < BROKEN_AREA) or (length < BROKEN_LENGTH) or (aspect_ratio < MIN_ASPECT_RATIO)

        if is_broken:
            color = (0, 0, 255)  # Red box for broken
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            cv2.drawContours(result_img, [box], 0, color, 2)
            broken_count += 1
        else:
            whole_count += 1

    # Draw summary at bottom-left
    summary = f"Broken: {broken_count}, Whole: {whole_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    (text_w, text_h), _ = cv2.getTextSize(summary, font, font_scale, thickness)
    margin = 10
    x = margin
    y = result_img.shape[0] - margin

    cv2.rectangle(result_img,
                  (x - 5, y - text_h - 5),
                  (x + text_w + 5, y + 5),
                  bg_color, -1)

    cv2.putText(result_img, summary, (x, y),
                font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

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
