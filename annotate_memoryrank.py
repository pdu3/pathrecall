import json
import os
import cv2

# 参数设置
jsonl_path = "memoryrank_candidates.jsonl"
image_base = "frames"
output_path = "memoryrank_labeled.jsonl"
max_candidates = 5

# 加载数据
with open(jsonl_path, "r") as f:
    samples = [json.loads(line) for line in f]

print(f"总计 {len(samples)} 条样本，开始标注...")

# 开始逐个标注
labeled_samples = []
for idx, sample in enumerate(samples):
    query_img_path = os.path.join(image_base, os.path.basename(sample["query_img"]))
    query = cv2.imread(query_img_path)
    query = cv2.resize(query, (300, 300))
    canvas = [query]

    # 显示候选图像
    candidate_imgs = []
    for cpath in sample["candidates"]:
        cimg_path = os.path.join(image_base, os.path.basename(cpath))
        cimg = cv2.imread(cimg_path)
        if cimg is None:
            cimg = 255 * np.ones((300, 300, 3), dtype=np.uint8)
        else:
            cimg = cv2.resize(cimg, (300, 300))
        candidate_imgs.append(cimg)

    combined = cv2.hconcat([query] + candidate_imgs)
    cv2.imshow(f"[{idx+1}/{len(samples)}] Query + Candidates", combined)
    print("⚙️ 输入与 query 最相关的候选编号（如 1 或 1,3），然后按 Enter：")
    print("候选图编号为：1~5（从左至右），Query 图总在最左侧。")

    key = input("相关候选图编号: ").strip()
    selected = set()
    for k in key.split(","):
        if k.strip().isdigit():
            i = int(k.strip()) - 1
            if 0 <= i < len(sample["candidates"]):
                selected.add(i)

    labels = [1 if i in selected else 0 for i in range(len(sample["candidates"]))]
    sample["labels"] = labels

    # 你可以在这里添加 sample["question"] 字段
    sample["question"] = input("请输入该 query 图像的问题（英文）: ").strip()

    labeled_samples.append(sample)
    cv2.destroyAllWindows()

    # 实时保存
    with open(output_path, "a") as out_f:
        out_f.write(json.dumps(sample) + "\n")

print("✅ 全部标注完成，保存至：", output_path)
