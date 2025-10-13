# NSFW Detector API

This project provides an **NSFW (Not Safe For Work) content detection API** that analyzes text extracted from images.  
It combines OCR, NLP preprocessing, and toxicity detection using the Detoxify model.

---

## 🧩 Project Structure

```
nsfw_detector/
├── .env                     # Environment variables
├── README.md                # Project documentation
├── pyproject.toml           # Project dependencies
├── Dockerfile               # Docker configuration
├── api.py                   # FastAPI entry point
├── data/
│   ├── nsfw_list.txt        # List of NSFW keywords
│   └── dataset_image_test/  # Example test images
└── src/nsfw_detector/
    ├── __init__.py
    ├── utils.py             # Image metadata extraction
    ├── ocr.py               # OCR text extraction
    ├── nlp.py               # Text cleaning and rewriting
    └── toxic.py             # Toxicity scoring and NSFW classification
```

---

## 🚀 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/nsfw_detector.git
cd nsfw_detector
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

or if you are using Poetry:

```bash
poetry install
```

### 3️⃣ Environment Variables

Create a `.env` file in the project root:

```
TOXIC_THRESHOLD=0.0105
NSFW_WORD_FILE=./data/nsfw_list.txt
DATA_DIR=./data/
```

---

## 🐳 Running with Docker

```bash
docker build -t nsfw-detector .
docker run -p 8000:8000 nsfw-detector
```

Access the API at:  
👉 **http://localhost:8000/docs**

---

## 🧠 Design Overview

This project transforms the original Jupyter notebook pipeline into a modular API.  
It performs the following steps:

1. **OCR Extraction (`ocr.py`)**
   - Uses PaddleOCR to extract text from images.
   - Reads font and NSFW metadata if available.

2. **Regex Cleaning (`nlp.py`)**
   - Removes non-alphanumeric characters.
   - Keeps punctuation, digits, and spaces.

3. **Spell Correction & Word Segmentation**
   - Uses `symspellpy` to fix typos and add missing spaces between words.

4. **NSFW Word Detection (`toxic.py`)**
   - Checks whether the text contains known NSFW words (from `nsfw_list.txt`).

5. **Toxicity Scoring**
   - Uses the `Detoxify` model (based on BERT) to predict NSFW probability.

6. **Final Classification**
   - Combines both signals (NSFW word detection + Detoxify score).
   - A text is classified as NSFW if either check returns `True`.

---

## 📦 API Usage

### Upload a Zip File

Use `/predict` endpoint to upload a zip file containing images:

```bash
curl -X POST "http://localhost:8000/predict"      -H "accept: application/json"      -H "Content-Type: multipart/form-data"      -F "file=@my_images.zip"
```

The zip file should have this structure:

```
my_images.zip
├── img1.jpg
├── img2.png
```

### Response Example

```json
[
  {"image_name": "img1.jpg", "nsfw_final": true},
  {"image_name": "img2.png", "nsfw_final": false}
]
```

---

## ⚙️ Development Notes

- The OCR, SymSpell, and Detoxify models are initialized once at startup for efficiency.
- The pipeline supports easy expansion to multi-core or GPU acceleration.
- To improve accuracy, thresholds in `.env` can be adjusted based on dataset characteristics.

---

## 🧑‍💻 Author

Developed by Célia ZHANG
Based on a custom notebook pipeline for image-based NSFW detection.