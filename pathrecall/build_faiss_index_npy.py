import os
import numpy as np
import faiss

feature_dir = "data/features"
index_save_path = "data/faiss/image_vectors.index"

# 读取所有图像特征
features = []
for fname in sorted(os.listdir(feature_dir)):
    if fname.endswith(".npy"):
        features.append(np.load(os.path.join(feature_dir, fname)))

features = np.stack(features).astype("float32")
print("🔍 实际加载的特征数量:", features.shape[0])  # ← 新加
# 归一化特征向量
faiss.normalize_L2(features)

# 构建 FAISS 索引（使用内积近似余弦相似度）
index = faiss.IndexFlatIP(features.shape[1])
index.add(features)

# 保存索引
os.makedirs(os.path.dirname(index_save_path), exist_ok=True)
faiss.write_index(index, index_save_path)

print(f"✅ 构建完成并保存向量数据库到: {index_save_path}")
print(f"🔎 特征维度: {features.shape[1]}, 向量数量: {features.shape[0]}")
