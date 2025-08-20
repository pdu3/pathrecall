# from transformers import Blip2Processor, Blip2VisionModel, Blip2Config
from PIL import Image
import torch, os, numpy as np
from transformers import AutoImageProcessor, AutoTokenizer
from transformers import Blip2Processor, Blip2Model

device = "cuda"
blip2 = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=False)

from transformers import AutoTokenizer, AutoImageProcessor
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=False)
image_processor = AutoImageProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
processor = Blip2Processor(tokenizer=tokenizer, image_processor=image_processor)

image_dir = "data/path_images"
output_dir = "data/features"
os.makedirs(output_dir, exist_ok=True)

skipped = []

for fname in os.listdir(image_dir):
    if not fname.endswith(".jpg"): continue

    npy_path = os.path.join(output_dir, fname.replace(".jpg", ".npy"))
    if os.path.exists(npy_path):
        continue  # 跳过已处理的图像
    try:
        image = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_output = blip2.vision_model(pixel_values=inputs.pixel_values)
            feat = vision_output.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()  # [1408]
        if feat.ndim == 1 and feat.shape[0] > 0:
            np.save(os.path.join(output_dir, fname.replace(".jpg", ".npy")), feat)
        else:
            skipped.append(fname)

        # np.save(npy_path, feat)
    except Exception:
        skipped.append((fname, str(e)))
    # np.save(os.path.join(output_dir, fname.replace(".jpg", ".npy")), feat)

print(f"✅ 完成特征提取，总共: {len(os.listdir(output_dir))} 个特征文件")

print(f"⚠️ 共跳过了 {len(skipped)} 个异常文件：")
if skipped:
    print("⚠️ 以下图像处理失败:")
    for name, err in skipped:
        print(f"{name} -> {err}")
