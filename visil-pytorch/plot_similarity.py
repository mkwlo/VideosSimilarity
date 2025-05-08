import json
import matplotlib.pyplot as plt

# Wczytaj plik z wynikami
with open("results_detailed.json", "r") as f:
    results = json.load(f)

# Przejdź przez każde porównanie
for result in results:
    query_id = result["video_query"]
    target_id = result["video_target"]
    suspicious_frames = result["suspicious_frames"]

    # Dane do wykresu
    frame_nums = [f["db_frame"] for f in suspicious_frames]
    sim_scores = [f["max_similarity_with_query"] for f in suspicious_frames]

    # Wykres
    plt.figure(figsize=(10, 4))
    plt.plot(frame_nums, sim_scores, marker='o', linestyle='-', color='blue')
    plt.title(f"Frame similarity: {target_id} vs {query_id}")
    plt.xlabel("Frame number (test video)")
    plt.ylabel("Max similarity to reference video frames")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
