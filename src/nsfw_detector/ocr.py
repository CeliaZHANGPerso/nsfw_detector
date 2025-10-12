from paddleocr import PaddleOCR

def init_ocr():
    """
    初始化 PaddleOCR 模型
    """
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    return ocr
