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
    h, w = img.shape[:2]

    # Step 1: Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 2: Mask light regions (white, yellowish rice)
    mask_light = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    mask_color = cv2.inRange(hsv, (10, 40, 80), (40, 150, 255))
    mask = cv2.bitwise_or(mask_light, mask_color)

    # Fallback: If nothing found, use adaptive threshold
    if np.sum(mask) < 5000:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

    # Step 3: Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    # Step 4: Distance transform + Watershed to split merged grains
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Connected components
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    # Step 5: Extract contours
    contours = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask_label = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours += cnts

    # Step 6: Filter and measure
    grains = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (width, height), angle = rect
        length = max(width, height)
        grains.append((rect, length))

    if not grains:
        cv2.putText(result_img, "No grains detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return result_img

    # Step 7: Classify relative
    lengths = sorted([length for _, length in grains], reverse=True)
    top_N = max(3, len(lengths) // 5)
    reference_length = np.median(lengths[:top_N])
    broken_threshold = 0.65 * reference_length

    broken_count = 0
    for rect, length in grains:
        is_broken = length < broken_threshold
        if is_broken:
            color = (0, 0, 255)  # Red only for broken
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(result_img, [box], 0, color, 2)
            broken_count += 1

    # Step 8: Annotate result — ONLY broken count
    summary = f"Broken: {broken_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(summary, font, 0.6, 2)
    cv2.rectangle(result_img,
                  (10, h - 10 - text_h - 10),
                  (10 + text_w + 10, h - 10),
                  (0, 0, 0), -1)
    cv2.putText(result_img, summary, (15, h - 15),
                font, 0.6, (255, 255, 255), 2)

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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)