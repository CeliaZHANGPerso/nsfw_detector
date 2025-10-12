from detoxify import Detoxify
import pandas as pd

def load_nsfw_words(filepath):
    with open(filepath) as f:
        words = f.read().splitlines()
    return set(words)

def contains_nsfw_word(text, nsfw_words):
    if not text:
        return False
    text_words = text.split()
    return any(word in nsfw_words for word in text_words)

def predict_toxicity(text, model):
    if not text:
        return 0.0
    result = model.predict(text)
    return result[max(result)]

def process_predictions(df, nsfw_words, detox_model, threshold=0.0105):
    df["pred_score"] = df["corr_text"].apply(lambda x: predict_toxicity(x, detox_model))
    df["nsfw_pred"] = df["pred_score"] > threshold
    df["nsfw_word"] = df["corr_text"].apply(lambda x: contains_nsfw_word(x, nsfw_words))
    df["nsfw_final"] = df["nsfw_pred"] | df["nsfw_word"]
    return df
