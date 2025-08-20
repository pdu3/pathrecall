import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel

# ========== 配置路径 ==========
IMAGE_FOLDER = "data/path_images"  # 图像文件夹
INDEX_SAVE_PATH = "faiss_index/image_vectors.index"
PATHS_JSON_PATH = "faiss_index/image_paths.json"

# ========== 模型加载 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ========== 图像向量提取函数 ==========
def extract_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features[0].cpu().numpy()

# ========== 主函数 ==========
def build_faiss_index():
    os.makedirs(os.path.dirname(INDEX_SAVE_PATH), exist_ok=True)
    image_paths = sorted([
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    features = []
    for path in tqdm(image_paths, desc="提取图像特征"):
        emb = extract_embedding(path)
        features.append(emb)

    features_np = np.stack(features).astype(np.float32)
    index = faiss.IndexFlatIP(features_np.shape[1])
    index.add(features_np)

    # 保存 index 和路径映射
    faiss.write_index(index, INDEX_SAVE_PATH)
    with open(PATHS_JSON_PATH, "w") as f:
        json.dump(image_paths, f)

    print(f"✅ FAISS 索引保存至 {INDEX_SAVE_PATH}")
    print(f"✅ 图像路径列表保存至 {PATHS_JSON_PATH}")

# ========== 执行 ==========
if __name__ == "__main__":
    build_faiss_index()
