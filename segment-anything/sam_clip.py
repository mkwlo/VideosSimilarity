import cv2
import torch
import numpy as np
import clip
import time
from PIL import Image
from torchvision.transforms import ToPILImage
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

# === CONFIG ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
RESIZE_WIDTH = 120  # Set to None to keep original size"cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
FRAME_SKIP = 5
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
VIDEO1 = "kangur1.mp4"
VIDEO2 = "kangur2.mp4"
SAVE_SEGMENT_VIS = True  # Enable to save segment visualizations

os.makedirs("segments", exist_ok=True)

torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

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

# === FUNCTIONS ===
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
    try:
        image = clip_preprocess(Image.fromarray(image_np)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = clip_model.encode_image(image).to(DEVICE)
        return embedding.cpu().numpy()[0]
    except:
        return None

def visualize_segments(frame, masks, frame_id, video_label):
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20')
    alpha = 0.5  # transparency

    vis = frame.astype(np.float32) / 255.0  # normalize to [0,1] for blending

    for i, m in enumerate(masks):
        mask = m['segmentation']
        if mask.sum() == 0:
            continue
        color_rgba = cmap(i % cmap.N)
        color_rgb = np.array(color_rgba[:3])  # RGB in [0,1]
        color_bgr = color_rgb[::-1]  # Convert RGB to BGR
        vis[mask] = (1 - alpha) * vis[mask] + alpha * color_bgr

    vis = (vis * 255).astype(np.uint8)  # convert back to uint8
    output_path = f"segments/{video_label}_frame_{frame_id}.jpg"
    cv2.imwrite(output_path, vis)

def get_embeddings_from_frame(frame, frame_id, video_label):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)

    if SAVE_SEGMENT_VIS:
        visualize_segments(frame, masks, frame_id, video_label)

    embeddings = []
    for m in masks:
        x, y, w, h = cv2.boundingRect(m['segmentation'].astype(np.uint8))
        crop = frame[y:y+h, x:x+w]
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            emb = get_clip_embedding(crop)
            if emb is not None:
                embeddings.append(emb)
    return embeddings

def get_video_basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def process_video(video_path, label=None):
    print(f"Processing video: {video_path}")
    start = time.time()
    video_basename = get_video_basename(video_path) if label is None else label
    frames = extract_frames(video_path, FRAME_SKIP)
    all_embeddings = []
    for idx, (true_idx, frame) in enumerate(frames):
        emb = get_embeddings_from_frame(frame, true_idx, video_basename)
        all_embeddings.extend(emb)
        print(f"Frame {true_idx} -> {len(emb)} segments")
    print(f"Processing time for '{video_path}': {time.time() - start:.2f} seconds")
    return np.array(all_embeddings)

def compare_embedding_sets(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    sim = cosine_similarity(set1, set2)
    return np.mean(sim)

# === MAIN ===
if __name__ == "__main__":
    start_total = time.time()
    emb1 = process_video(VIDEO1)
    emb2 = process_video(VIDEO2)
    similarity = compare_embedding_sets(emb1, emb2)
    print(f"\nSimilarity between videos: {similarity:.4f}")
    print(f"\nTotal analysis time: {time.time() - start_total:.2f} seconds")
