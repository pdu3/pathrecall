import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ===== 参数配置 =====
VIDEO_DIR = "videos"                       # 输入视频文件夹路径（支持多个视频）
OUTPUT_ROOT = "data/path_images"          # 所有图片统一保存到这个目录
FRAME_INTERVAL = 2                         # 每隔多少秒抽一帧
SSIM_THRESHOLD = 0.90                      # 相似度阈值，避免保存重复帧
BLURRY_THRESHOLD = 100.0                   # 模糊判定阈值（Laplacian 方差）

# ===== 判断两帧是否相似 =====
def is_similar(img1, img2, threshold=SSIM_THRESHOLD):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score > threshold

# ===== 判断图像是否模糊 =====
def is_blurry(img, threshold=BLURRY_THRESHOLD):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold

# ===== 抽帧主逻辑 =====
def extract_keyframes(video_path, output_dir, interval_sec):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)

    saved_count = 0
    last_saved = None
    frame_id = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_prefix = os.path.splitext(os.path.basename(video_path))[0]
    pbar = tqdm(total=total_frames, desc=f"处理 {video_prefix}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            if is_blurry(frame):
                pbar.set_description("跳过模糊帧")
            elif last_saved is None or not is_similar(frame, last_saved):
                out_path = os.path.join(output_dir, f"{video_prefix}_frame_{saved_count:04d}.jpg")
                if not os.path.exists(out_path):  # 避免覆盖旧图
                    cv2.imwrite(out_path, frame)
                last_saved = frame
                saved_count += 1
                pbar.set_description(f"保存: {os.path.basename(out_path)}")

        frame_id += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"✅ [{video_prefix}] 共保存 {saved_count} 张关键帧 -> {output_dir}")

# ===== 执行入口（多视频遍历） =====
if __name__ == "__main__":
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    for video_name in video_files:
        video_path = os.path.join(VIDEO_DIR, video_name)
        extract_keyframes(video_path, OUTPUT_ROOT, FRAME_INTERVAL)


# import cv2
# import os
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
# from tqdm import tqdm

# # ===== 参数配置 =====
# VIDEO_DIR = "videos"                       # 输入视频文件夹路径（支持多个视频）
# OUTPUT_ROOT = "data/path_images"          # 输出根目录
# FRAME_INTERVAL = 2                        # 每隔多少秒抽一帧
# SSIM_THRESHOLD = 0.90                     # 相似度阈值，避免保存重复帧
# BLURRY_THRESHOLD = 100.0                  # 模糊判定阈值（Laplacian 方差）

# # ===== 判断两帧是否相似 =====
# def is_similar(img1, img2, threshold=SSIM_THRESHOLD):
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     score, _ = ssim(gray1, gray2, full=True)
#     return score > threshold

# # ===== 判断图像是否模糊 =====
# def is_blurry(img, threshold=BLURRY_THRESHOLD):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return lap_var < threshold

# # ===== 抽帧主逻辑 =====
# def extract_keyframes(video_path, output_dir, interval_sec):
#     os.makedirs(output_dir, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = int(fps * interval_sec)

#     saved_count = 0
#     last_saved = None
#     frame_id = 0

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     pbar = tqdm(total=total_frames, desc=f"处理 {os.path.basename(video_path)}")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_id % frame_interval == 0:
#             if is_blurry(frame):
#                 pbar.set_description("跳过模糊帧")
#             elif last_saved is None or not is_similar(frame, last_saved):
#                 out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
#                 cv2.imwrite(out_path, frame)
#                 last_saved = frame
#                 saved_count += 1
#                 pbar.set_description(f"保存: {out_path}")

#         frame_id += 1
#         pbar.update(1)

#     cap.release()
#     pbar.close()
#     print(f"✅ [{os.path.basename(video_path)}] 共保存 {saved_count} 张关键帧 -> {output_dir}")

# # ===== 执行入口（多视频遍历） =====
# if __name__ == "__main__":
#     video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

#     for video_name in video_files:
#         video_path = os.path.join(VIDEO_DIR, video_name)
#         # output_dir = os.path.join(OUTPUT_ROOT, os.path.splitext(video_name)[0])
#         output_dir = OUTPUT_ROOT
#         extract_keyframes(video_path, output_dir, FRAME_INTERVAL)

# import cv2
# import os
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
# from tqdm import tqdm

# # ===== 配置参数 =====
# VIDEO_PATH = "sample.mp4"        # 输入视频路径
# OUTPUT_DIR = "data/path_images"        # 输出图像保存路径
# FRAME_INTERVAL = 3                     # 每隔多少秒抽一帧
# SSIM_THRESHOLD = 0.90                  # 超过这个相似度就跳过（图像太像）

# # ===== 判断两帧是否相似 =====
# def is_similar(img1, img2, threshold=SSIM_THRESHOLD):
#     img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     score, _ = ssim(img1_gray, img2_gray, full=True)
#     return score > threshold

# # ===== 抽帧主逻辑 =====
# def extract_keyframes(video_path, output_dir, interval_sec):
#     os.makedirs(output_dir, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = int(fps * interval_sec)

#     saved_count = 0
#     last_saved = None
#     frame_id = 0

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     pbar = tqdm(total=total_frames, desc="抽帧中")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_id % frame_interval == 0:
#             if last_saved is None or not is_similar(frame, last_saved):
#                 out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
#                 cv2.imwrite(out_path, frame)
#                 last_saved = frame
#                 saved_count += 1

#         frame_id += 1
#         pbar.update(1)

#     cap.release()
#     pbar.close()
#     print(f"✅ 共保存 {saved_count} 张关键帧，保存在: {output_dir}")

# # ===== 执行入口 =====
# if __name__ == "__main__":
#     extract_keyframes(VIDEO_PATH, OUTPUT_DIR, FRAME_INTERVAL)
