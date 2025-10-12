import os
import asyncio
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import torch

# detoxify uses transformers + torch internally
from detoxify import Detoxify

_toxic_model = None
_toxic_executor = None

def init_toxic_model(device: str = "cpu"):
    global _toxic_model, _toxic_executor
    if _toxic_model is None:
        _toxic_model = Detoxify('unbiased')  # name as in prior code
        # Detoxify encapsulates model internals; ensure device placement if possible
        try:
            if device == "cuda" and torch.cuda.is_available():
                # internal models may use device automatically; no easy direct move
                pass
        except Exception:
            pass
    if _toxic_executor is None:
        _toxic_executor = ThreadPoolExecutor(max_workers=2)
    return _toxic_model

def predict_toxicity_single(model, text: str) -> float:
    if not text:
        return 0.0
    # Detoxify predict accepts string or list; returns dict of arrays if list
    try:
        res = model.predict(text)
    except Exception:
        # fallback to empty
        return 0.0
    # the model returns dict with keys like 'toxicity', etc. Choose best relevant key
    # We try to get 'toxicity' or max over keys
    if isinstance(res, dict):
        # if values are floats or arrays
        values = []
        for v in res.values():
            try:
                if isinstance(v, (list, tuple)):
                    values.append(float(v[0]))
                else:
                    values.append(float(v))
            except Exception:
                continue
        if values:
            return max(values)
    return 0.0

def predict_toxicity_batch(model, texts: List[str], batch_size: int = 8) -> List[float]:
    """
    Batch predict toxicity scores. This runs synchronously but should be called from executor if needed.
    """
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            res = model.predict(batch)
            # res may be dict of lists
            if isinstance(res, dict):
                # choose 'toxicity' if present else max over keys
                if 'toxicity' in res:
                    scores.extend([float(x) for x in res['toxicity']])
                else:
                    # compute max across keys per item
                    n = len(next(iter(res.values())))
                    for idx in range(n):
                        vals = []
                        for v in res.values():
                            try:
                                vals.append(float(v[idx]))
                            except Exception:
                                continue
                        scores.append(max(vals) if vals else 0.0)
            else:
                # fallback: zeros
                scores.extend([0.0]*len(batch))
        except Exception:
            scores.extend([0.0]*len(batch))
    return scores

async def predict_toxicity_async(model, texts: List[str], batch_size: int = 8):
    """
    Async wrapper that runs batch predict in a thread pool to avoid blocking event loop.
    """
    loop = asyncio.get_event_loop()
    global _toxic_executor
    if _toxic_executor is None:
        _toxic_executor = ThreadPoolExecutor(max_workers=1)
    return await loop.run_in_executor(_toxic_executor, lambda: predict_toxicity_batch(model, texts, batch_size))

def contains_nsfw_word(text: str, nsfw_words: List[str]) -> bool:
    if not text:
        return False
    words = set([w.strip().lower() for w in text.split()])
    for w in nsfw_words:
        if w.lower() in words:
            return True
    return False

def combine_nsfw(pred_score: float, nsfw_word_flag: bool, threshold: float = 0.5) -> bool:
    return (pred_score >= threshold) or nsfw_word_flag
