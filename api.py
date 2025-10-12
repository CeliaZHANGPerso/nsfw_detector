from fastapi import FastAPI
from src.nsfw_detector.ocr import init_ocr
from src.nsfw_detector.utils import extract_image_metadata_to_df
from src.nsfw_detector.nlp import clean_text, load_symspell, rewrite_text
from src.nsfw_detector.toxic import load_nsfw_words, process_predictions
from detoxify import Detoxify
import pandas as pd
from dotenv import load_dotenv
import os

app = FastAPI(title="NSFW Detector API")

TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", 0.0105))
NSFW_WORD_FILE = os.getenv("NSFW_WORD_FILE", "./data/nsfw_list.txt")

# initialize models
ocr_model = init_ocr()
sym_spell = load_symspell()
detox_model = Detoxify("unbiased")
nsfw_words = load_nsfw_words(NSFW_WORD_FILE)

@app.get("/predict")
def predict_nsfw():
    image_dir = "data/dataset_image_test"
    df = extract_image_metadata_to_df(image_dir, ocr_model)
    df["re_text"] = df["extracted_text"].apply(clean_text)
    df["corr_text"] = df["re_text"].apply(lambda x: rewrite_text(x, sym_spell))
    df = process_predictions(df, nsfw_words, detox_model, threshold=TOXIC_THRESHOLD)
    return df[["image_name", "nsfw_final"]].to_dict(orient="records")
