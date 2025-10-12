import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import pkg_resources

# prefer wordninja for fast deglue; fallback to wordsegment if needed
try:
    import wordninja
    _USE_WORDNINJA = True
except Exception:
    _USE_WORDNINJA = False
    from wordsegment import load as ws_load, segment as ws_segment
    ws_load()

# SymSpell
try:
    from symspellpy import SymSpell
    _HAS_SYMSPELL = True
except Exception:
    _HAS_SYMSPELL = False

_symspell = None
_rewrite_executor = None

def init_symspell(prefix_length: int = 7, max_edit: int = 2):
    global _symspell, _rewrite_executor
    if not _HAS_SYMSPELL:
        return None
    if _symspell is None:
        _symspell = SymSpell(max_dictionary_edit_distance=max_edit, prefix_length=prefix_length)
        dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        _symspell.load_dictionary(dict_path, term_index=0, count_index=1)
    if _rewrite_executor is None:
        _rewrite_executor = ThreadPoolExecutor(max_workers=4)
    return _symspell

def clean_text(text: str) -> str:
    if not text:
        return ""
    # 保留字母 数字 常用标点 空格
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\:\;\'\"/\-\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def deglue_text(text: str) -> str:
    """
    Handle stuck-together words like "Thegraphiccontentwarningshould"
    Try wordninja first, else use wordsegment
    """
    if not text:
        return ""
    s = text.strip()
    if _USE_WORDNINJA:
        return " ".join(wordninja.split(s.lower()))
    else:
        return " ".join(ws_segment(s.lower()))

async def rewrite_text_async(sym_spell, text: str):
    """
    Use SymSpell segmentation & correction asynchronously if available;
    otherwise return cleaned + deglued text.
    """
    if not text:
        return ""
    loop = asyncio.get_event_loop()
    if sym_spell is None:
        # just do deglue
        return deglue_text(clean_text(text))
    # run symspell word segmentation in thread pool
    def _call():
        seg = sym_spell.word_segmentation(text.lower())
        return seg.corrected_string
    global _rewrite_executor
    if _rewrite_executor is None:
        _rewrite_executor = ThreadPoolExecutor(max_workers=4)
    corrected = await loop.run_in_executor(_rewrite_executor, _call)
    return corrected
