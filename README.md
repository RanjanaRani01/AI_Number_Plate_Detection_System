# 🚘 AI Number Plate Detection & Recognition System

An AI-powered Automatic Number Plate Recognition (ANPR) system built using **YOLOv8, EasyOCR, Tesseract OCR, OpenCV, and Streamlit**. The system detects vehicle number plates from images, videos, and live webcam feeds, then extracts and validates the plate number with high accuracy.

## ✨ Features

* YOLOv8-based Number Plate Detection
* OCR using EasyOCR + Tesseract OCR Fusion
* Image, Video, and Webcam Support
* Indian Number Plate Validation
* Vehicle Type Classification (Private, Commercial, Electric, Rental)
* Confidence Score Display
* CSV Export of Detection Results
* User-Friendly Streamlit Interface

## 🛠️ Tech Stack

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* EasyOCR
* Tesseract OCR
* Streamlit
* NumPy & Pandas

## 📂 Project Structure

```text
├── app.py
├── train_model.py
├── split_dataset.py
├── data.yaml
├── best.pt
├── requirements.txt
└── README.md
```

## ⚙️ Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/AI-Number-Plate-Recognition-System.git
cd AI-Number-Plate-Recognition-System
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR separately (not included in this repository).

After installation, update the Tesseract path in `app.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 📸 Screenshots

<img width="1806" height="958" alt="Screenshot 2026-06-01 121150" src="https://github.com/user-attachments/assets/688bfbe1-651b-4424-86bf-6f6ebb9620cd" />
<img width="1784" height="1003" alt="Screenshot 2026-06-01 121128" src="https://github.com/user-attachments/assets/634f6772-eaad-425c-88e7-42d4a8bae0c7" />
<img width="1822" height="970" alt="Screenshot 2026-06-01 120929" src="https://github.com/user-attachments/assets/65f71e65-3784-406f-afe2-d01721e7ee3b" />

## 👨‍💻 Author
**Ranjana Rani**
AI/ML Enthusiast | Machine Learning | Computer Vision
