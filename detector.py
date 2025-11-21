import os
import insightface

MODEL_PATH = os.path.join("models", "scrfd_10g_bnkps.onnx")
assert os.path.exists(MODEL_PATH), f"Файл модели не найден: {MODEL_PATH}"

detector = insightface.model_zoo.get_model(MODEL_PATH)
detector.prepare(ctx_id=-1, input_size=(640, 640))  #CPU

def detect_faces(img, min_conf=0.3):

    try:
        bboxes, landmarks = detector.detect(img)
        filtered = [(bbox, lm) for bbox, lm in zip(bboxes, landmarks) if bbox[4] >= min_conf]
        if filtered:
            bboxes_f, landmarks_f = zip(*filtered)
            return list(bboxes_f), list(landmarks_f)
        else:
            return [], []
    except Exception as e:
        print(f"Ошибка при детекции лиц: {e}")
        return [], []
