import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import faiss
from transformers import Blip2Processor, Blip2Model

# === 配置路径 ===
IMAGE_FOLDER = "data/path_images"
INDEX_SAVE_PATH = "faiss_index/image_vectors.index"
PATHS_JSON_PATH = "faiss_index/image_paths.json"

# === 模型加载 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
blip2 = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")

# === 提取 patch token 均值向量 ===
def extract_blip2_feat(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_output = blip2.vision_model(pixel_values=inputs.pixel_values)
        patch_tokens = vision_output.last_hidden_state  # [1, T, 768]
        pooled = patch_tokens.mean(dim=1).squeeze(0).cpu().numpy()  # [768]
        return pooled

# === 构建 FAISS 索引 ===
def build_faiss_index():
    os.makedirs(os.path.dirname(INDEX_SAVE_PATH), exist_ok=True)
    image_paths = sorted([
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    features = []
    for path in tqdm(image_paths, desc="提取图像特征（BLIP-2 patch token mean）"):
        vec = extract_blip2_feat(path)
        features.append(vec)

    features_np = np.stack(features).astype(np.float32)
    faiss.normalize_L2(features_np)  # 归一化向量以使用内积作为余弦相似度

    index = faiss.IndexFlatIP(features_np.shape[1])  # Index for cosine similarity
    index.add(features_np)

    faiss.write_index(index, INDEX_SAVE_PATH)
    with open(PATHS_JSON_PATH, "w") as f:
        json.dump(image_paths, f)

    print(f"✅ FAISS 索引保存至 {INDEX_SAVE_PATH}")
    print(f"✅ 图像路径列表保存至 {PATHS_JSON_PATH}")
    print(f"🔎 向量维度: {features_np.shape[1]}")

# === 执行 ===
if __name__ == "__main__":
    build_faiss_index()
