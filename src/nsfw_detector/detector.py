import os
import asyncio

from typing import List
from dotenv import load_dotenv
import tempfile

from nsfw_detector.ocr import init_ocr, ocr_image_async, init_thread_pool
from nsfw_detector.nlp import init_symspell, clean_text, rewrite_text_async
from nsfw_detector.toxic import init_toxic_model, predict_toxicity_async, contains_nsfw_word, combine_nsfw

# load env
load_dotenv(".env")
DEVICE = os.getenv("DEVICE", "cpu")
OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))
REWRITE_WORKERS = int(os.getenv("REWRITE_WORKERS", "4"))
TOXIC_BATCH_SIZE = int(os.getenv("TOXIC_BATCH_SIZE", "8"))
TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", "0.5"))

# initialize resources
use_gpu = (DEVICE == "cuda")
ocr_engine = init_ocr(use_gpu=use_gpu)
init_thread_pool(OCR_WORKERS)
sym_spell = init_symspell()
toxic_model = init_toxic_model(device=DEVICE)

# load nsfw wordlist
NSFW_WORD_FILE = "data/nsfw_list.txt"
nsfw_words = []
if os.path.exists(NSFW_WORD_FILE):
    with open(NSFW_WORD_FILE, "r", encoding="utf-8") as f:
        nsfw_words = [l.strip() for l in f if l.strip()]
else:
    nsfw_words = []

# concurrency limiter
SEM = asyncio.Semaphore(OCR_WORKERS + REWRITE_WORKERS)

async def process_single_image(file: UploadFile):
    try:
        content = await file.read()
        # save to temp file because PaddleOCR expects file path or numpy array
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        async with SEM:
            # OCR
            texts = await ocr_image_async(ocr_engine, tmp_path, max_workers=OCR_WORKERS)
        # join extracted lines
        raw_text = " ".join(texts).strip()
        cleaned = clean_text(raw_text)
        # rewrite / deglue & correct (async)
        rewritten = await rewrite_text_async(sym_spell, cleaned)
        # whether contains nsfw word by list
        word_flag = contains_nsfw_word(rewritten, nsfw_words)

        # return intermediate info (score will be filled after batch)
        return {
            "filename": file.filename,
            "raw_text": raw_text,
            "cleaned": cleaned,
            "rewritten": rewritten,
            "word_flag": word_flag
        }
    except Exception as e:
        return {"filename": getattr(file, "filename", "unknown"), "error": str(e)}


async def predict_nsfw(files: List[UploadFile] = File(...)):
    """
    Accept multiple files, process OCR+rewrite concurrently (async),
    then batch predict toxicity scores for all rewritten texts,
    then compute final nsfw verdict combining word_flag OR tox_score >= threshold.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # 1) process images concurrently to produce rewritten texts
    tasks = [process_single_image(f) for f in files]
    intermediates = await asyncio.gather(*tasks)

    # collect texts for batch toxicity prediction
    texts = [item.get("rewritten", "") if item.get("rewritten", None) is not None else "" for item in intermediates]
    # 2) batch toxicity predict (async wrapper)
    scores = await predict_toxicity_async(toxic_model, texts, batch_size=TOXIC_BATCH_SIZE)

    # 3) assemble results
    results = []
    for item, score in zip(intermediates, scores):
        if "error" in item:
            results.append({
                "filename": item.get("filename"),
                "error": item.get("error")
            })
            continue
        wf = item.get("word_flag", False)
        final_flag = combine_nsfw(score, wf, threshold=TOXIC_THRESHOLD)
        results.append({
            "filename": item.get("filename"),
            "raw_text": item.get("raw_text"),
            "rewritten": item.get("rewritten"),
            "score": float(score),
            "contains_nsfw_word": bool(wf),
            "final_nsfw": bool(final_flag)
        })
    return JSONResponse({"predictions": results})