import cv2
import torch
import clip
import numpy as np
from torchvision.transforms import ToPILImage
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG ===
FRAME_SKIP = 5  # Co ile klatek przetwarzać
YOLO_MODEL = "yolov5s.pt"

# === LOAD MODELS ===
device = "cuda" if torch.cuda.is_available() else "cpu"
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
        if frame_idx % step == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames

def detect_objects(frame):
    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy()  # x1, y1, x2, y2, conf, class
    return detections

def crop_objects(frame, detections):
    crops = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
    return crops

def get_clip_embeddings(crops):
    embeddings = []
    for crop in crops:
        try:
            img = ToPILImage()(crop[:, :, ::-1])  # BGR to RGB
            img_pre = clip_preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clip_model.encode_image(img_pre).cpu().numpy()[0]
            embeddings.append(emb)
        except:
            continue
    return embeddings

def process_video(video_path):
    frames = extract_frames(video_path, FRAME_SKIP)
    all_embeddings = []
    for frame in frames:
        detections = detect_objects(frame)
        crops = crop_objects(frame, detections)
        embeddings = get_clip_embeddings(crops)
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)

def compare_videos(video1, video2):
    emb1 = process_video(video1)
    emb2 = process_video(video2)
    if len(emb1) == 0 or len(emb2) == 0:
        return 0.0
    similarity = cosine_similarity(emb1, emb2)
    return np.mean(similarity)

# === MAIN ===
if __name__ == "__main__":
    video_path1 = "kangur1.mp4"
    video_path2 = "kangur2.mp4"
    score = compare_videos(video_path1, video_path2)
    print(f"Similarity between videos: {score:.4f}")
