import cv2
def draw_faces(image, bboxes, labels=None, ids=None):
    img = image.copy()

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox[:4])
        color = (0, 255, 0)  # зелёная рамка по умолчанию

        #Цвет для шума
        if labels and labels[i] == -1:
            color = (0, 0, 255)  # красный

        #Нарисовать рамку
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        #Подпись
        if ids:
            label_text = ids[i]
        elif labels:
            label_text = "unknown" if labels[i] == -1 else f"person_{labels[i]+1}"
        else:
            label_text = "face"

        #Написать текст
        cv2.putText(img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return img

