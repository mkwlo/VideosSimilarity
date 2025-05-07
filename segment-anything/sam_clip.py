import cv2
import torch
import numpy as np
import clip
from PIL import Image
from torchvision.transforms import ToPILImage
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
FRAME_SKIP = 30
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
VIDEO1 = "kangur1.mp4"
VIDEO2 = "kangur2.mp4"

torch.backends.cudnn.benchmark = True  # optimize for consistent input sizes

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append(frame)
        frame_idx += 1
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

def get_embeddings_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)

    embeddings = []
    for m in masks:
        x, y, w, h = cv2.boundingRect(m['segmentation'].astype(np.uint8))
        crop = frame[y:y+h, x:x+w]
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            emb = get_clip_embedding(crop)
            if emb is not None:
                embeddings.append(emb)
    return embeddings

def process_video(video_path):
    print(f"Processing video: {video_path}")
    frames = extract_frames(video_path, FRAME_SKIP)
    all_embeddings = []
    for idx, frame in enumerate(frames):
        emb = get_embeddings_from_frame(frame)
        all_embeddings.extend(emb)
        print(f"Frame {idx} -> {len(emb)} segments")
    return np.array(all_embeddings)

def compare_embedding_sets(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    sim = cosine_similarity(set1, set2)
    return np.mean(sim)

# === MAIN ===
if __name__ == "__main__":
    emb1 = process_video(VIDEO1)
    emb2 = process_video(VIDEO2)
    similarity = compare_embedding_sets(emb1, emb2)
    print(f"\n🔍 Similarity between videos: {similarity:.4f}")
