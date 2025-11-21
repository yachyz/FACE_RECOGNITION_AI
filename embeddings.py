import os
import numpy as np
import insightface
from insightface.app.common import Face

MODEL_PATH = os.path.join("models", "w600k_r50.onnx")
assert os.path.exists(MODEL_PATH), f"Файл модели не найден: {MODEL_PATH}"

#ctx_id=-1 CPU, ctx_id=0 GPU
embedder = insightface.model_zoo.get_model(MODEL_PATH)
embedder.prepare(ctx_id=-1)

def get_embedding(img, bbox, landmark=None):

    if landmark is None:
        return None

    face = Face()
    face.bbox = bbox
    face.kps = landmark

    try:
        embedding = embedder.get(img, face)
        if embedding is None:
            return None
        embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"Ошибка при получении эмбеддинга: {e}")
        return None
