import os
import torch
import faiss
import json
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import Blip2Processor, Blip2Model
from typing import List

# === Ensure model is downloaded ===
os.environ["TRANSFORMERS_CACHE"] = "./.cache"

# === Load BLIP-2 ===
blip2 = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl").to("cuda")
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=False)

from transformers import AutoTokenizer, AutoImageProcessor
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=False)
image_processor = AutoImageProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
processor = Blip2Processor(tokenizer=tokenizer, image_processor=image_processor)

# === Cross-Attention Fusion ===
class CrossAttentionFusion(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, memory):
        attn_output, _ = self.cross_attn(query, memory, memory)
        return query + attn_output  # Residual connection

# === Extract patch tokens from BLIP2 vision encoder ===
def extract_patch_tokens(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        vision_output = blip2.vision_model(pixel_values=inputs.pixel_values)
        return vision_output.last_hidden_state  # [1, T, 768]

# === FAISS Retrieval ===
def retrieve_top_k(current_feat_np, index, image_paths, k=3, threshold=0.9, min_required=2):
    D, I = index.search(current_feat_np.astype(np.float32), k)
    filtered = [(i, d) for i, d in zip(I[0], D[0]) if d >= threshold]
    if len(filtered) < min_required:
        fallback = I[0][:min_required]
        return [image_paths[i] for i in fallback]
    return [image_paths[i] for i, _ in filtered]

# === Full pipeline ===
def run_memory_injection_pipeline(current_image_path: str, question: str,
                                   fusion_module: CrossAttentionFusion,
                                   faiss_index,
                                   image_paths: list):
    # 1. Encode current image
    current_feat = extract_patch_tokens(current_image_path)  # [1, T, 768]

    # 2. Mean pool for FAISS
    pooled_feat = current_feat.mean(dim=1).detach().cpu().numpy()  # [1, 768]
    retrieved_paths = retrieve_top_k(pooled_feat, faiss_index, image_paths, k=3, threshold=0.9, min_required=2)

    # 3. Encode memory images
    memory_feats = []
    for p in retrieved_paths:
        feat = extract_patch_tokens(p)  # [1, T, 768]
        memory_feats.append(feat.squeeze(0))  # [T, 768]
    memory_stack = torch.cat(memory_feats, dim=0).unsqueeze(0).to("cuda")  # [1, T_all, 768]

    # 4. Fuse
    fused_feat = fusion_module(current_feat, memory_stack)  # [1, T, 768]

    print(f"Fused feature shape: {fused_feat.shape}")
    return fused_feat

# === Main Test ===
# if __name__ == "__main__":
#     faiss_index = faiss.read_index("faiss_index/image_vectors.index")
#     with open("faiss_index/image_paths.json") as f:
#         image_paths = json.load(f)  # expects ["path1.jpg", "path2.jpg", ...]

#     fusion = CrossAttentionFusion(dim=1408, num_heads=8).to("cuda")

#     fused_feat = run_memory_injection_pipeline(
#         current_image_path="query_images/query.jpg",
#         question="Did I just pass a red chair?",
#         fusion_module=fusion,
#         faiss_index=faiss_index,
#         image_paths=image_paths
#     )
