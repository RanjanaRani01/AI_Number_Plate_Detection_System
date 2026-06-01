# =========================================================
# 🚘 ULTRA PRO AI NUMBER PLATE SYSTEM (FINAL FIXED)
# =========================================================

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import easyocr
import pytesseract
import re
import os
import tempfile
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="NUMBER PLATE DETECTION",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================
MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)

reader = easyocr.Reader(['en'], gpu=True)

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =========================================================
# CREATE FOLDER
# =========================================================
os.makedirs("plates", exist_ok=True)

# =========================================================
# SESSION
# =========================================================
if "data" not in st.session_state:
    st.session_state.data = []

if "best" not in st.session_state:
    st.session_state.best = {}

# =========================================================
# UI
# =========================================================
st.title("🚘 AI NUMBER PLATE RECOGINITION SYSTEM")

# =====confidence slider=======

conf = st.sidebar.slider(
    "Confidence",
    0.05,
    0.9,
    0.15
)

imgsz = st.sidebar.selectbox(
    "Image Size",
    [640, 960, 1280, 1600, 1920],
    index=4
)

frame_skip = st.sidebar.slider(
    "Frame Skip",
    1,
    20,
    10
)

# =========================================================
# SMART CLEAN
# =========================================================
def clean(text):

    text = text.upper()

    text = re.sub(r'[^A-Z0-9]', '', text)

    chars = list(text)

    for i in range(len(chars)):

        if i < 2:

            if chars[i] == '0':
                chars[i] = 'O'

        else:

            if chars[i] == 'O':
                chars[i] = '0'

            elif chars[i] == 'I':
                chars[i] = '1'

            elif chars[i] == 'Z':
                chars[i] = '2'

            elif chars[i] == 'S':
                chars[i] = '5'

    return "".join(chars)

# =========================================================
# VALIDATE INDIAN PLATE
# =========================================================
def validate_plate(t):

    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'

    return re.match(pattern, t) is not None

# =========================================================
# SMART VEHICLE TYPE DETECTION
# =========================================================
def vehicle_type(img):

    try:

        # Resize
        plate = cv2.resize(
            img,
            (300, 100)
        )

        hsv = cv2.cvtColor(
            plate,
            cv2.COLOR_BGR2HSV
        )

        # =========================================
        # WHITE MASK
        # =========================================
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 120]),
            np.array([180, 60, 255])
        )

        # =========================================
        # YELLOW MASK
        # =========================================
        yellow_mask = cv2.inRange(
            hsv,
            np.array([15, 80, 80]),
            np.array([40, 255, 255])
        )

        # =========================================
        # GREEN MASK
        # =========================================
        green_mask = cv2.inRange(
            hsv,
            np.array([40, 40, 40]),
            np.array([90, 255, 255])
        )

        # =========================================
        # BLACK MASK
        # =========================================
        black_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 0]),
            np.array([180, 255, 50])
        )

        # =========================================
        # PIXELS
        # =========================================
        white_pixels = cv2.countNonZero(white_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        black_pixels = cv2.countNonZero(black_mask)

        total = plate.shape[0] * plate.shape[1]

        white_ratio = white_pixels / total
        yellow_ratio = yellow_pixels / total
        green_ratio = green_pixels / total
        black_ratio = black_pixels / total

        # =========================================
        # FINAL TYPE
        # =========================================
        if white_ratio > 0.35:
            return "🚗 Private"

        elif yellow_ratio > 0.25:
            return "🚚 Commercial"

        elif green_ratio > 0.20:
            return "⚡ Electric"

        elif black_ratio > 0.40:
            return "🚕 Rental"

        return "Unknown"

    except:
        return "Unknown"

# =========================================================
# IMAGE ENHANCE
# =========================================================
def enhance(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Denoise
    gray = cv2.bilateralFilter(
        gray,
        11,
        17,
        17
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    gray = clahe.apply(gray)

    return gray

# =========================================================
# OCR
# =========================================================
def ocr_fusion(img):

    try:

        img = cv2.resize(
            img,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        gray = enhance(img)

        blur = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        if blur < 20:
            return ""

        processed = []

        # =========================================
        # OTSU
        # =========================================
        _, th1 = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        processed.append(th1)

        # =========================================
        # INVERSE
        # =========================================
        _, th2 = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        processed.append(th2)

        # =========================================
        # ADAPTIVE
        # =========================================
        th3 = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2
        )

        processed.append(th3)

        # =========================================
        # SHARPEN
        # =========================================
        kernel = np.array([
            [0, -1, 0],
            [-1, 5,-1],
            [0, -1, 0]
        ])

        sharp = cv2.filter2D(
            gray,
            -1,
            kernel
        )

        processed.append(sharp)

        all_results = []

        config = (
            '--oem 3 --psm 7 '
            '-c tessedit_char_whitelist='
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        for p in processed:

            # =====================================
            # TESSERACT
            # =====================================
            text = pytesseract.image_to_string(
                p,
                config=config
            )

            text = clean(text)

            if len(text) >= 8:
                all_results.append(text)

            # =====================================
            # EASYOCR
            # =====================================
            try:

                result = reader.readtext(p)

                for r in result:

                    txt = clean(r[1])

                    if len(txt) >= 8:
                        all_results.append(txt)

            except:
                pass

        valid = []

        for t in all_results:

            t = t.replace("IND", "")

            t = re.sub(
                r'[^A-Z0-9]',
                '',
                t
            )

            if validate_plate(t):
                valid.append(t)

        if valid:

            best = max(
                set(valid),
                key=valid.count
            )

            return best

        return ""

    except:
        return ""

# =========================================================
# DETECT
# =========================================================
def detect(frame):

    original = frame.copy()

    # ============================================
    # NO RESIZE
    # ============================================
    small = frame.copy()

    h_ratio = 1
    w_ratio = 1

    # ============================================
    # YOLO
    # ============================================
    results = model.predict(
        source=small,
        conf=0.15,
        imgsz=1920,
        iou=0.4,
        augment=True,
        verbose=False
    )

    best_box = None
    best_conf = 0

    for r in results:

        if r.boxes is None:
            continue

        for b in r.boxes:

            x1, y1, x2, y2 = map(
                int,
                b.xyxy[0]
            )

            x1 = int(x1 * w_ratio)
            x2 = int(x2 * w_ratio)
            y1 = int(y1 * h_ratio)
            y2 = int(y2 * h_ratio)

            c = float(b.conf[0])

            crop = original[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # =====================================
            # SMALL PLATE SUPPORT
            # =====================================
            if crop.shape[0] < 15 or crop.shape[1] < 40:
                continue

            # =====================================
            # OCR
            # =====================================
            text = ocr_fusion(crop)

            if not validate_plate(text):
                continue

            if c > best_conf:

                best_conf = c

                best_box = (
                    x1,
                    y1,
                    x2,
                    y2,
                    text,
                    crop,
                    c
                )

    # ============================================
    # DRAW
    # ============================================
    if best_box:

        x1, y1, x2, y2, text, crop, c = best_box

        vtype = vehicle_type(crop)

        if (
            text not in st.session_state.best or
            c > st.session_state.best[text]
        ):

            st.session_state.best[text] = c

            cv2.imwrite(
                f"plates/{text}.jpg",
                crop
            )

            st.session_state.data.append({
                "Plate": text,
                "Vehicle": vtype,
                "Confidence": round(c * 100, 2)
            })

        # ========================================
        # BOX
        # ========================================
        cv2.rectangle(
            original,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # ========================================
        # LABEL
        # ========================================
        label = f"{text} | {vtype} | {round(c * 100,1)}%"

        cv2.putText(
            original,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return original

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "🖼 Image",
    "🎥 Video",
    "📷 Webcam"
])

# =========================================================
# IMAGE
# =========================================================
with tab1:

    f = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"]
    )

    if f:

        img = Image.open(f)

        frame = np.array(img)

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        out = detect(frame)

        st.image(
            cv2.cvtColor(
                out,
                cv2.COLOR_BGR2RGB
            )
        )

# =========================================================
# VIDEO
# =========================================================
with tab2:

    v = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if v:

        t = tempfile.NamedTemporaryFile(
            delete=False
        )

        t.write(v.read())

        cap = cv2.VideoCapture(t.name)

        frame_box = st.empty()

        frame_count = 0

        progress = st.progress(0)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        current = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            current += 1
            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            out = detect(frame)

            frame_box.image(
                cv2.cvtColor(
                    out,
                    cv2.COLOR_BGR2RGB
                ),
                channels="RGB"
            )

            progress.progress(
                min(current / total_frames, 1.0)
            )

        cap.release()

        st.success("✅ Video Processing Completed")

# =========================================================
# WEBCAM
# =========================================================
with tab3:

    run = st.checkbox("Start Webcam")

    frame_box = st.image([])

    cap = cv2.VideoCapture(0)

    frame_count = 0

    while run:

        ret, frame = cap.read()

        if not ret:
            st.error("Camera Error")
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        out = detect(frame)

        frame_box.image(
            cv2.cvtColor(
                out,
                cv2.COLOR_BGR2RGB
            ),
            channels="RGB"
        )

    cap.release()

# =========================================================
# RESULTS
# =========================================================
st.subheader("📊 Detection Results")

if st.session_state.data:

    df = pd.DataFrame(
        st.session_state.data
    )

    df = df.drop_duplicates(
        subset=["Plate"],
        keep="last"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(
        index=False
    ).encode('utf-8')

    st.download_button(
        "⬇ Download CSV",
        csv,
        "plates.csv",
        "text/csv"
    )

else:
    st.info("No detections yet")

# =========================================================
# CLEAR
# =========================================================
if st.button("🗑 Clear Data"):

    st.session_state.data = []

    st.session_state.best = {}

    st.success("Data Cleared")