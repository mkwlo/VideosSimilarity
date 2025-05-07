import cv2
import torch
import clip
import numpy as np
from torchvision.transforms import ToPILImage
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.cm as cm

# === CONFIG ===
FRAME_SKIP = 5
RESIZE_WIDTH = 512
YOLO_MODEL = "yolov5s.pt"
N_CLUSTERS = 5

# === LOAD MODELS ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO(YOLO_MODEL)

# === FUNCTIONS ===
def extract_frames(video_path, step=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if RESIZE_WIDTH is not None:
            h, w = frame.shape[:2]
            scale = RESIZE_WIDTH / w
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        if frame_idx % step == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames

def detect_boxes(frame):
    results = yolo_model.predict(frame, device=0 if device == "cuda" else "cpu")
    boxes = results[0].boxes.xyxy.cpu().numpy()
    return boxes

def crop_objects(frame, boxes):
    crops = []
    coords = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
            coords.append((x1, y1, x2, y2))
    return crops, coords

def get_clip_embeddings(crops):
    embeddings = []
    for crop in crops:
        try:
            img = ToPILImage()(crop[:, :, ::-1])
            img_pre = clip_preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clip_model.encode_image(img_pre).cpu().numpy()[0]
            embeddings.append(emb)
        except:
            continue
    return embeddings

def color_by_cluster(cluster_id):
    cmap = cm.get_cmap("tab10")
    color = cmap(cluster_id % 10)[:3]  # RGB
    return tuple(int(c * 255) for c in color[::-1])  # BGR for OpenCV

def process_video_for_embeddings(video_path):
    frames = extract_frames(video_path, FRAME_SKIP)
    all_embeddings = []
    all_coords = []
    frame_refs = []
    os.makedirs("highlighted_frames", exist_ok=True)

    for i, frame in enumerate(frames):
        frame_idx = i * FRAME_SKIP
        boxes = detect_boxes(frame)
        crops, coords = crop_objects(frame, boxes)
        embeddings = get_clip_embeddings(crops)

        all_embeddings.extend(embeddings)
        all_coords.extend(coords)
        frame_refs.extend([(frame_idx, frame.copy()) for _ in coords])

    if not all_embeddings:
        return np.array([])

    km = KMeans(n_clusters=min(N_CLUSTERS, len(all_embeddings)), random_state=0).fit(all_embeddings)
    labels = km.labels_

    # draw
    frames_colored = {}
    for (frame_idx, base_frame), (x1, y1, x2, y2), cluster in zip(frame_refs, all_coords, labels):
        color = color_by_cluster(cluster)
        if frame_idx not in frames_colored:
            frames_colored[frame_idx] = base_frame.copy()
        cv2.rectangle(frames_colored[frame_idx], (x1, y1), (x2, y2), color, 2)

    for idx, img in frames_colored.items():
        out_path = f"highlighted_frames/{os.path.splitext(os.path.basename(video_path))[0]}_frame{idx}.jpg"
        cv2.imwrite(out_path, img)

    return np.array(all_embeddings)

import time

# === MAIN ===
if __name__ == "__main__":
    total_start = time.time()
    video_path1 = "kangur1.mp4"
    video_path2 = "kangur2.mp4"
    emb1 = process_video_for_embeddings(video_path1)
    emb2 = process_video_for_embeddings(video_path2)

    if len(emb1) == 0 or len(emb2) == 0:
        print("No embeddings found in one or both videos.")
    else:
        sim_matrix = cosine_similarity(emb1, emb2)
        mean_sim = np.mean(sim_matrix)
        print(f"Similarity between videos: {mean_sim:.4f}")
        print(f"Total analysis time: {time.time() - total_start:.2f} seconds")
