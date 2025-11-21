import os
import cv2
import json
from collections import defaultdict
from detector import detect_faces
from embeddings import get_embedding
from clustering import cluster_embeddings
from visualization import draw_faces
from utils import make_person_ids, save_face_crop

#Конфигурация проекта
CONFIG = {
    "input_dir": "images",
    "output_dir": "results",
    "clustering": {
        "method": "dbscan",       #"dbscan" или "kmeans"
        "eps": 0.6,               #радиус соседства для DBSCAN
        "min_samples": 2,         #минимальное число точек для DBSCAN
        "metric": "cosine",       #метрика для DBSCAN
        "n_clusters": 5           #число кластеров для KMeans
    },
    "detector": {
        "min_conf": 0.3           #порог уверенности для детектора лиц
    }
}


def main():
    print(f"Запуск пайплайна: обработка изображений из {CONFIG['input_dir']} сохранение в {CONFIG['output_dir']}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    all_embeddings = []
    all_bboxes = []
    all_images = []
    image_files = [f for f in os.listdir(CONFIG["input_dir"]) if f.lower().endswith((".jpg", ".png"))]

    #Детекция и эмбеддинги
    for fname in image_files:
        path = os.path.join(CONFIG["input_dir"], fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Не удалось прочитать изображение: {path}")
            continue

        print(f"\nОбрабатываю: {fname}")
        bboxes, landmarks = detect_faces(img, min_conf=CONFIG["detector"]["min_conf"])
        print(f"Найдено лиц: {len(bboxes)}")

        for bbox, lm in zip(bboxes, landmarks):
            emb = get_embedding(img, bbox, lm)
            if emb is not None:
                all_embeddings.append(emb)
                all_bboxes.append(bbox)
                all_images.append((fname, img))
            else:
                print(f"Эмбеддинг не получен для лица: {bbox}")

    #Кластеризация
    print(f"\nКластеризация {len(all_embeddings)} эмбеддингов...")
    all_labels = cluster_embeddings(
        all_embeddings,
        eps=CONFIG["clustering"]["eps"],
        min_samples=CONFIG["clustering"]["min_samples"],
        metric=CONFIG["clustering"]["metric"],
        method=CONFIG["clustering"]["method"],
        n_clusters=CONFIG["clustering"]["n_clusters"]
    )

    #Присвоение идентификаторов
    person_ids = make_person_ids(all_labels)

    #Визуализация и сохранение
    grouped = defaultdict(list)
    for (fname, img), bbox, label, pid in zip(all_images, all_bboxes, all_labels, person_ids):
        grouped[fname].append((img, bbox, label, pid))

    results_json = []
    for fname in grouped:
        img = grouped[fname][0][0]
        bboxes = [x[1] for x in grouped[fname]]
        labels = [x[2] for x in grouped[fname]]
        ids = [x[3] for x in grouped[fname]]

        out_img = draw_faces(img, bboxes, labels, ids)
        out_path = os.path.join(CONFIG["output_dir"], f"result_{fname}")
        cv2.imwrite(out_path, out_img)

        for bbox, label, pid in zip(bboxes, labels, ids):
            results_json.append({
                "file": fname,
                "bbox": bbox[:4].tolist() if hasattr(bbox, "tolist") else list(map(int, bbox[:4])),
                "label": int(label),
                "person_id": pid
            })
            #сохраняем кропы лиц по папкам
            save_face_crop(img, bbox, pid)

    #Сохранение JSON
    json_path = os.path.join(CONFIG["output_dir"], "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Результаты сохранены в {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()