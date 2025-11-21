import os
import cv2
import numpy as np

def make_person_ids(labels):
    ids = []
    cluster_map = {}
    next_id = 1

    for label in labels:
        if label == -1:
            ids.append(f"person_{next_id}")
            next_id += 1
        else:
            if label not in cluster_map:
                cluster_map[label] = f"person_{next_id}"
                next_id += 1
            ids.append(cluster_map[label])

    return ids


def crop_face(image, bbox, margin=0.1):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox[:4])

    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)

    return image[y1:y2, x1:x2]


def save_face_crop(image, bbox, person_id, output_dir="results/crops"):

    os.makedirs(output_dir, exist_ok=True)
    person_dir = os.path.join(output_dir, person_id)
    os.makedirs(person_dir, exist_ok=True)

    face_crop = crop_face(image, bbox)
    filename = f"{person_id}_{np.random.randint(10000)}.jpg"
    path = os.path.join(person_dir, filename)
    cv2.imwrite(path, face_crop)
