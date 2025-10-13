from fastapi import FastAPI, UploadFile, File
from src.nsfw_detector.ocr import init_ocr
from src.nsfw_detector.utils import extract_image_metadata_to_df
from src.nsfw_detector.nlp import clean_text, load_symspell, rewrite_text
from src.nsfw_detector.toxic import load_nsfw_words, process_predictions
from detoxify import Detoxify
import pandas as pd
from dotenv import load_dotenv
import tempfile
import zipfile
import os

app = FastAPI(title="NSFW Detector API")

TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", 0.006435))
NSFW_WORD_FILE = os.getenv("NSFW_WORD_FILE", "./data/nsfw_list.txt")
DATA_DIR = os.getenv("DATA_DIR", "./data/")

# initialize models
ocr_model = init_ocr()
sym_spell = load_symspell()
detox_model = Detoxify("unbiased")
nsfw_words = load_nsfw_words(NSFW_WORD_FILE)
data_dir = DATA_DIR

async def extract_zip_to_unzipdir(file: UploadFile) -> str:
    """
    unzip uploaded zip file to a temporary directory and return the path
    """
    content = await file.read()
    zip_path = os.path.join(data_dir, file.filename)

    with open(zip_path, "wb") as f:
        f.write(content)

    # unzip
    unzip_dir = os.path.join(data_dir, "unzipped")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(unzip_dir)

    # find extracted directory (assuming single top-level dir)
    extracted_dirs = [
        os.path.join(unzip_dir, d)
        for d in os.listdir(unzip_dir)
        if os.path.isdir(os.path.join(unzip_dir, d))
    ]

    if extracted_dirs:
        return extracted_dirs[1]


def analyze_image_folder(image_dir: str) -> pd.DataFrame:
    """
    NLP part of the pipeline: OCR -> Clean -> Rewrite -> Predict
    """
    df = extract_image_metadata_to_df(image_dir, ocr_model)

    if "extracted_text" not in df.columns or df.empty:
        raise ValueError("No images with extracted text found in the provided zip file.")

    df["re_text"] = df["extracted_text"].apply(clean_text)
    df["corr_text"] = df["re_text"].apply(lambda x: rewrite_text(x, sym_spell))
    df = process_predictions(df, nsfw_words, detox_model, threshold=0.5)

    return df[["image_name", "nsfw_final"]]


@app.post("/predict")
async def predict_nsfw(file: UploadFile = File(...)):
    """
    upload a zip file containing images, return a list of dicts with image names and NSFW predictions.
    """
    try:
        image_dir = await extract_zip_to_unzipdir(file)
        df_result = analyze_image_folder(image_dir)
        return df_result.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}
