import os
import asyncio
from typing import List, Dict, Any
from PIL import Image
import concurrent.futures
import multiprocessing

def list_image_files(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def pil_open_safe(path_or_bytes):
    from io import BytesIO
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return Image.open(BytesIO(path_or_bytes)).convert("RGB")
    else:
        return Image.open(path_or_bytes).convert("RGB")
