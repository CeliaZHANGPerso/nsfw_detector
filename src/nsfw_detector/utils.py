import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

def extract_image_metadata_to_df(image_dir, ocr_model):
    """
    从 dataset_image_test 文件夹中提取图片文字、字体、nsfw元数据等。
    """
    records = []
    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            img = Image.open(img_path)
            metadata = img.info
            font = metadata.get("font", "Unknown")
            nsfw = metadata.get("nsfw", "Unknown")
            text_all = ocr_model.predict(img_path)
            text = " ".join(text_all[0]['rec_texts'])
            records.append({
                "image_name": img_name,
                "extracted_text": text,
                "font": font,
                "nsfw": nsfw
            })
        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {e}")
            records.append({
                "image_name": img_name,
                "extracted_text": "",
                "font": "Unknown",
                "nsfw": "Unknown"
            })
    return pd.DataFrame(records)
