import cv2
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === KONFIGURACJA ===
NUM_CLASSES = 21
FRAME_SKIP = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === MODELE ===
model = deeplabv3_mobilenet_v3_large(pretrained=True).to(device).eval()
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((480, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# === FUNKCJE ===
def extract_frames(video_path, step=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames

def segment_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).cpu().numpy().astype(np.uint8)
    return mask

def get_class_histogram(mask):
    hist = np.bincount(mask.flatten(), minlength=NUM_CLASSES)
    return hist / hist.sum() if hist.sum() > 0 else hist

def process_video_semantics(video_path):
    frames = extract_frames(video_path, FRAME_SKIP)
    histograms = []
    for frame in frames:
        mask = segment_frame(frame)
        hist = get_class_histogram(mask)
        histograms.append(hist)
    return np.mean(histograms, axis=0)

def compare_histograms(hist1, hist2):
    return cosine_similarity([hist1], [hist2])[0][0]

def find_least_similar_frame(reference_hist, video_path):
    frames = extract_frames(video_path, FRAME_SKIP)
    worst_similarity = 1.0
    worst_frame_idx = -1
    worst_frame = None
    for i, frame in enumerate(frames):
        mask = segment_frame(frame)
        hist = get_class_histogram(mask)
        sim = compare_histograms(reference_hist, hist)
        print(f"Klatka {i}: similarity = {sim:.4f}")
        if sim < worst_similarity:
            worst_similarity = sim
            worst_frame_idx = i
            worst_frame = frame
    return worst_frame_idx, worst_similarity, worst_frame

# === MAIN ===
if __name__ == "__main__":
    video1 = "kangur1.mp4"
    video2 = "kangur2.mp4"

    print("▶️ Przetwarzanie kangur1 (referencja)...")
    hist1 = process_video_semantics(video1)

    print("\n🔍 Szukanie najmniej podobnej klatki w kangur2...")
    idx, sim, frame = find_least_similar_frame(hist1, video2)

    if frame is not None:
        out_path = "najmniej_podobna_klatka.jpg"
        cv2.imwrite(out_path, frame)
        print(f"\n🟥 Najmniej podobna klatka to #{idx}, similarity = {sim:.4f}")
        print(f"🖼️ Zapisano jako: {out_path}")
    else:
        print("❌ Nie znaleziono klatek.")
