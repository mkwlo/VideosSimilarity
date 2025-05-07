import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator
from evaluation import extract_features


def calculate_frame_to_frame_similarities(model, query_feat, db_feat, args):
    """
    Zwraca macierz podobieństwa [query_frame x db_frame].
    """
    with torch.no_grad():
        q = query_feat.to(args.gpu_id)  # [Q, D1, D2]
        t = db_feat.to(args.gpu_id)    # [T, D1, D2]

        q = q.mean(dim=1)  # [Q, 512]
        t = t.mean(dim=1)  # [T, 512]

        sim_matrix = torch.nn.functional.cosine_similarity(
            q.unsqueeze(1),  # [Q, 1, 512]
            t.unsqueeze(0),  # [1, T, 512]
            dim=2
        )  # [Q, T]

        return sim_matrix.cpu().numpy()


def frame_to_seconds(frame_idx, fps):
    return round(int(frame_idx) / fps, 2)


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='Video similarity frame-by-frame using ViSiL.', formatter_class=formatter)
    parser.add_argument('--query_file', type=str, required=True, help='Path to file that contains the query videos')
    parser.add_argument('--database_file', type=str, required=True, help='Path to file that contains the database videos')
    parser.add_argument('--output_file', type=str, default='results_detailed.json', help='Output file name')
    parser.add_argument('--batch_sz', type=int, default=128, help='Batch size for feature extraction')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for dataloaders')
    parser.add_argument('--fps', type=float, default=25.0, help='Frames per second of the video')
    parser.add_argument('--input_fps', type=int, default=25, help='Frame rate used when loading videos')
    args = parser.parse_args()

    # Inicjalizacja modelu
    model = ViSiL(pretrained=True, symmetric=True).to(args.gpu_id)
    model.eval()

    # Ładowanie zapytań
    query_generator = VideoGenerator(args.query_file, fps=args.input_fps)
    query_loader = DataLoader(query_generator, num_workers=args.workers)

    queries, queries_ids = [], []
    print('> Extracting features from query videos')
    for video in tqdm(query_loader):
        frames = video[0][0]
        video_id = video[1][0]
        print(f"[INFO] Loaded {frames.shape[0]} frames for query video {video_id}")
        features = extract_features(model, frames, args).cpu()
        queries.append(features)
        queries_ids.append(video_id)

    # Ładowanie bazy
    db_generator = VideoGenerator(args.database_file, fps=args.input_fps)
    db_loader = DataLoader(db_generator, num_workers=args.workers)

    # Porównania
    all_detailed_results = []
    print('\n> Calculating frame-to-frame similarities')
    for db_video in tqdm(db_loader):
        db_frames = db_video[0][0]
        db_video_id = db_video[1][0]
        print(f"[INFO] Loaded {db_frames.shape[0]} frames for database video {db_video_id}")

        if db_frames.shape[0] > 1:
            db_features = extract_features(model, db_frames, args).cpu()

            for i, q_feat in enumerate(queries):
                sim_matrix = calculate_frame_to_frame_similarities(model, q_feat, db_features, args)
                sim_matrix = np.squeeze(sim_matrix)

                print(f"[DEBUG] query shape: {q_feat.shape}, db shape: {db_features.shape}, sim_matrix shape: {sim_matrix.shape}")

                max_sim_per_db_frame = np.max(sim_matrix, axis=0)  # Najbardziej podobna klatka z query dla każdej db-klatki

                threshold = np.percentile(max_sim_per_db_frame, 25)  # np. dolne 25% jako podejrzane
                suspicious_frames = []
                for dbf, score in enumerate(max_sim_per_db_frame):
                    if score < threshold:
                        suspicious_frames.append({
                            "db_frame": dbf,
                            "db_time_sec": frame_to_seconds(dbf, args.fps),
                            "max_similarity_with_query": float(score)
                        })

                per_video_result = {
                    "video_query": queries_ids[i],
                    "video_target": db_video_id,
                    "suspicious_frames": suspicious_frames
                }
                all_detailed_results.append(per_video_result)

    # Zapis do pliku
    with open(args.output_file, 'w') as f:
        json.dump(all_detailed_results, f, indent=2)

    print(f'Wyniki zapisane w {args.output_file}')
