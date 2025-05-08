import numpy as np
from scipy.spatial.distance import cdist
import itertools
import csv

class SimpleObjectTracker:
    def __init__(self, max_distance=0.4, max_frames_lost=5):
        self.objects = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost

    def update(self, detections, frame_id, csv_writer=None, video_label="video"):
        matched_ids = set()
        updated_objects = {}

        # Prepare embeddings and centroids for matching
        new_embeddings = [d['embedding'] for d in detections]
        new_centroids = [d['centroid'] for d in detections]

        if not self.objects:
            for det in detections:
                det['id'] = self.next_id
                updated_objects[self.next_id] = {
                    'embedding': det['embedding'],
                    'centroid': det['centroid'],
                    'last_seen': 0
                }
                if csv_writer:
                    csv_writer.writerow([video_label, frame_id, self.next_id] + list(det['centroid']))
                self.next_id += 1
            self.objects = updated_objects
            return detections

        current_ids = list(self.objects.keys())
        existing_embeddings = [self.objects[obj_id]['embedding'] for obj_id in current_ids]
        existing_centroids = [self.objects[obj_id]['centroid'] for obj_id in current_ids]

        # Compare new to existing by embedding distance
        dist_matrix = cdist(new_embeddings, existing_embeddings, metric='cosine')

        assigned = set()
        for i, row in enumerate(dist_matrix):
            min_idx = np.argmin(row)
            min_dist = row[min_idx]
            if min_dist < self.max_distance:
                matched_id = current_ids[min_idx]
                detections[i]['id'] = matched_id
                updated_objects[matched_id] = {
                    'embedding': detections[i]['embedding'],
                    'centroid': detections[i]['centroid'],
                    'last_seen': 0
                }
                matched_ids.add(matched_id)
                assigned.add(i)
                if csv_writer:
                    csv_writer.writerow([video_label, frame_id, matched_id] + list(detections[i]['centroid']))

        # Add new objects for unassigned
        for i in range(len(detections)):
            if i not in assigned:
                detections[i]['id'] = self.next_id
                updated_objects[self.next_id] = {
                    'embedding': detections[i]['embedding'],
                    'centroid': detections[i]['centroid'],
                    'last_seen': 0
                }
                if csv_writer:
                    csv_writer.writerow([video_label, frame_id, self.next_id] + list(detections[i]['centroid']))
                self.next_id += 1

        # Age unmatched objects and remove old
        for obj_id, obj in self.objects.items():
            if obj_id not in matched_ids:
                obj['last_seen'] += 1
                if obj['last_seen'] <= self.max_frames_lost:
                    updated_objects[obj_id] = obj

        self.objects = updated_objects
        return detections
