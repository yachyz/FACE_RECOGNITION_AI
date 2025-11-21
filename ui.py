import streamlit as st
import cv2
import numpy as np
import json
import os
import zipfile
import io
from detector import detect_faces
from embeddings import get_embedding
from clustering import cluster_embeddings
from visualization import draw_faces
from utils import make_person_ids, save_face_crop

st.set_page_config(
    page_title="Face Recognition",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Скрытие лишних элементов
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Поиск и группировка лиц")

uploaded_files = st.file_uploader(
    "Загрузите фото",
    type=["jpg", "png","jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    all_embeddings, all_bboxes, all_images = [], [], []

    #Прогресс выполнения
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        with st.spinner(f"Обработка {uploaded_file.name}..."):
            bboxes, landmarks = detect_faces(img, min_conf=0.3)
            for bbox, lm in zip(bboxes, landmarks):
                emb = get_embedding(img, bbox, lm)
                if emb is not None:
                    all_embeddings.append(emb)
                    all_bboxes.append(bbox)
                    all_images.append((uploaded_file.name, img))

        #обновляем прогресс
        progress = int((i + 1) / len(uploaded_files) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Обработано {i + 1} из {len(uploaded_files)} файлов")

    if all_embeddings:
        labels = cluster_embeddings(all_embeddings, method="dbscan", eps=0.6, min_samples=2)
        person_ids = make_person_ids(labels)

        st.success(f"Найдено {len(all_bboxes)} лиц, сгруппировано в {len(set(person_ids))} персонажей")

        results_json = []
        grouped_by_person = {}
        vis_dir = "results/visualized"
        os.makedirs(vis_dir, exist_ok=True)

        #Визуализация и сбор JSON
        NUM_COLUMNS = 6
        for (fname, img), bbox, label, pid in zip(all_images, all_bboxes, labels, person_ids):
            grouped_by_person.setdefault(pid, []).append({
                'file': fname,
                'img': img,
                'bbox': bbox,
                'label': label
            })

        for person_id, faces_data in grouped_by_person.items():
            st.subheader(f"{person_id} ({len(faces_data)} фото)")
            columns = st.columns(NUM_COLUMNS)
            for i, face_data in enumerate(faces_data):
                out_img = draw_faces(
                    face_data['img'],
                    [face_data['bbox']],
                    labels=[face_data['label']],
                    ids=[person_id]
                )
                col_index = i % NUM_COLUMNS
                with columns[col_index]:
                    st.image(
                        cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB),
                        caption=f"{face_data['file']}",
                        width=180
                    )

                results_json.append({
                    "file": face_data['file'],
                    "bbox": list(map(int, face_data['bbox'][:4])),
                    "label": int(face_data['label']),
                    "person_id": person_id
                })
                save_face_crop(face_data['img'], face_data['bbox'], person_id)

                save_path = os.path.join(vis_dir, f"{person_id}_{i}_{face_data['file']}")
                cv2.imwrite(save_path, out_img)

        #Сохраняем JSON
        os.makedirs("results", exist_ok=True)
        json_path = "results/results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = f.read()

        #Создаём ZIP архив
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for root, _, files in os.walk(vis_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=file)
        zip_buffer.seek(0)

        #Кнопки сверху
        st.download_button("⬇️ Скачать результаты (JSON)", json_data, "results.json", "application/json")
        st.download_button("⬇️ Скачать фото с визуализацией (ZIP)", zip_buffer, "visualized_faces.zip", "application/zip")

        #Таскбар (sidebar)
        st.sidebar.title("Навигация")
        st.sidebar.info(f"👤 {len(all_bboxes)} лиц, {len(set(person_ids))} персонажей")
        st.sidebar.download_button("⬇️ Скачать JSON", json_data, "results.json", "application/json")
        st.sidebar.download_button("⬇️ Скачать ZIP", zip_buffer, "visualized_faces.zip", "application/zip")

    else:
        st.error("Лица не найдены!")
