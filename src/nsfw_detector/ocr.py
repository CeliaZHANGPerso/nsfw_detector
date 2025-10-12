import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from paddleocr import PaddleOCR
from typing import Any, Dict, List, Tuple

# Global thread pool. We'll initialize with desired workers from env in api.
_thread_pool: ThreadPoolExecutor = None

def init_thread_pool(max_workers: int = 4):
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _thread_pool

def init_ocr(use_gpu: bool = False, lang: str = "en"):
    """
    Initialize PaddleOCR, use_gpu: bool
    """
    ocr = PaddleOCR(use_angle_cls=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    lang=lang,
                    use_gpu=use_gpu)
    return ocr

async def ocr_image_async(ocr, path_or_bytes, max_workers: int = 4):
    """
    Run OCR in threadpool. Return list of recognized lines as strings.
    """
    loop = asyncio.get_event_loop()
    pool = init_thread_pool(max_workers)
    def _call():
        # PaddleOCR returns nested structure: result[0] is list of lines, each line [bbox, (text, confidence)]
        res = ocr.ocr(path_or_bytes, cls=False)
        texts = []
        if res and len(res) > 0:
            # res is list per page; collect all
            for line in res[0]:
                try:
                    texts.append(line[1][0])
                except Exception:
                    continue
        return texts
    texts = await loop.run_in_executor(pool, _call)
    return texts
