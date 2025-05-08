import cv2
import torch
import numpy as np
import clip
import time
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import csv
from tracker import SimpleObjectTracker

# === CONFIG ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE_WIDTH = 400
FRAME_SKIP = 5
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
VIDEO1 = "kangur1.mp4"
VIDEO2 = "kangur2.mp4"
SAVE_SEGMENT_VIS = True
os.makedirs("segments", exist_ok=True)

# === LOAD MODELS ===
print("Loading SAM and CLIP models...")
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=512
)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.to(DEVICE)

def extract_frames(video_path, step=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    real_frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if RESIZE_WIDTH is not None:
            h, w = frame.shape[:2]
            scale = RESIZE_WIDTH / w
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        if frame_idx % step == 0:
            frames.append((real_frame_idx, frame))
        frame_idx += 1
        real_frame_idx += 1
    cap.release()
    return frames

def get_clip_embedding(image_np):
    image = clip_preprocess(Image.fromarray(image_np)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = clip_model.encode_image(image).to(DEVICE)
    return embedding.cpu().numpy()[0]

def visualize_segments(frame, detections, frame_id, video_label):
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20')
    alpha = 0.5
    vis = frame.astype(np.float32) / 255.0
    for det in detections:
        mask = det['mask']
        color_rgba = cmap(det['id'] % cmap.N)
        color_rgb = np.array(color_rgba[:3])
        color_bgr = color_rgb[::-1]
        vis[mask] = (1 - alpha) * vis[mask] + alpha * color_bgr
        cy, cx = map(int, det['centroid'])
        cv2.putText(vis, str(det['id']), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    vis = (vis * 255).astype(np.uint8)
    cv2.imwrite(f"segments/{video_label}_frame_{frame_id}.jpg", vis)

def get_embeddings_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)
    detections = []
    for m in masks:
        mask = m['segmentation']
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        crop = frame[y:y+h, x:x+w]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        emb = get_clip_embedding(crop)
        if emb is not None:
            M = cv2.moments(mask.astype(np.uint8))
            if M['m00'] == 0:
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            detections.append({
                'embedding': emb,
                'centroid': (cy, cx),
                'mask': mask
            })
    return detections

def get_video_basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def process_video(video_path, label=None):
    print(f"Processing video: {video_path}")
    start = time.time()
    video_basename = get_video_basename(video_path) if label is None else label
    frames = extract_frames(video_path, FRAME_SKIP)
    tracker = SimpleObjectTracker()
    all_embeddings = []

    with open(f"{video_basename}_tracking.csv", mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["video", "frame", "id", "centroid_y", "centroid_x"])
        for idx, (true_idx, frame) in enumerate(frames):
            detections = get_embeddings_from_frame(frame)
            tracked = tracker.update(detections, true_idx, csv_writer=writer, video_label=video_basename)
            all_embeddings.extend([d['embedding'] for d in tracked])
            if SAVE_SEGMENT_VIS:
                visualize_segments(frame, tracked, true_idx, video_basename)
            print(f"Frame {true_idx} -> {len(tracked)} tracked segments")

    print(f"Processing time for '{video_path}': {time.time() - start:.2f} seconds")
    return np.array(all_embeddings)

def compare_embedding_sets(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    sim = cosine_similarity(set1, set2)
    return np.mean(sim)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    start_total = time.time()
    emb1 = process_video(VIDEO1)
    emb2 = process_video(VIDEO2)
    similarity = compare_embedding_sets(emb1, emb2)
    print(f"\nSimilarity between videos: {similarity:.4f}")
    print(f"\nTotal analysis time: {time.time() - start_total:.2f} seconds")
