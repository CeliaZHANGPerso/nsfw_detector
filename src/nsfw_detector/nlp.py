import re
import pkg_resources
from symspellpy import SymSpell
import importlib
bigram_path = importlib.resources.files("symspellpy") / "frequency_bigramdictionary_en_243_342.txt"

def clean_text(text):
    """
    正则清理文本，只保留字母、数字、标点和空格
    """
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"/\-()]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_symspell():
    """
    初始化 SymSpell 拼写修正器
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

def rewrite_text(text, sym_spell):
    """
    拼写修正 + 分词
    """
    if not text:
        return ""
    else:
        result = sym_spell.word_segmentation(text.lower()).corrected_string
    return result
