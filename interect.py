import streamlit as st
import cv2
import numpy as np
import math
import tempfile
import pandas as pd
import os
import zipfile
import shutil
from PIL import Image

try:
    from streamlit_cropper import st_cropper
except ImportError:
    st.error("Пожалуйста, установите библиотеку: pip install streamlit-cropper")
    st.stop()

# --- НАСТРОЙКИ СТРАНИЦЫ И ДИЗАЙН ---
st.set_page_config(page_title="Physics Tracker Pro", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        border: 2px solid #ff4b4b;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff4b4b;
        color: white;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #2e7b32;
        color: white;
        border: none;
        padding: 15px;
        font-size: 18px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        color: #ff4b4b;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)


# --- ФИЗИЧЕСКИЙ ТРЕКЕР ---
class PhysicsTracker:
    def __init__(self):
        self.particles = {}
        self.id_count = 1
        self.max_lost = 10
        self.min_hits = 3

    def update(self, detections, frame_idx):
        predicted_positions = {}
        for internal_id, data in self.particles.items():
            px, py = data['pos']
            vx, vy = data['vel']
            predicted_positions[internal_id] = [px + vx, py + vy]

        result_boxes = []
        matched_internal_ids = set()

        for det in detections:
            x, y, w, h, det_cx, det_cy, area = det
            best_internal_id = None
            min_dist = float('inf')

            for internal_id, pred_pos in predicted_positions.items():
                if internal_id in matched_internal_ids:
                    continue
                dist = math.hypot(det_cx - pred_pos[0], det_cy - pred_pos[1])
                if dist < 80 and dist < min_dist:
                    min_dist = dist
                    best_internal_id = internal_id

            if best_internal_id is not None:
                old_cx, old_cy = self.particles[best_internal_id]['pos']
                new_vx = det_cx - old_cx
                new_vy = det_cy - old_cy
                v_smooth = 0.5
                smooth_vx = (1 - v_smooth) * self.particles[best_internal_id]['vel'][0] + v_smooth * new_vx
                smooth_vy = (1 - v_smooth) * self.particles[best_internal_id]['vel'][1] + v_smooth * new_vy

                self.particles[best_internal_id]['pos'] = [det_cx, det_cy]
                self.particles[best_internal_id]['vel'] = [smooth_vx, smooth_vy]
                self.particles[best_internal_id]['lost_frames'] = 0
                self.particles[best_internal_id]['hits'] += 1

                if self.particles[best_internal_id]['hits'] == self.min_hits:
                    if self.particles[best_internal_id]['display_id'] is None:
                        self.particles[best_internal_id]['display_id'] = self.id_count
                        self.id_count += 1

                matched_internal_ids.add(best_internal_id)
                disp_id = self.particles[best_internal_id]['display_id']
                if disp_id is not None:
                    result_boxes.append([x, y, w, h, disp_id, det_cx, det_cy, area])
            else:
                temp_key = f"cand_{frame_idx}_{len(matched_internal_ids)}"
                self.particles[temp_key] = {
                    'pos': [det_cx, det_cy],
                    'vel': [0, 0],
                    'lost_frames': 0,
                    'hits': 1,
                    'display_id': None
                }

        ids_to_delete = []
        for internal_id in self.particles.keys():
            if internal_id not in matched_internal_ids:
                self.particles[internal_id]['lost_frames'] += 1
                self.particles[internal_id]['pos'][0] += self.particles[internal_id]['vel'][0]
                self.particles[internal_id]['pos'][1] += self.particles[internal_id]['vel'][1]
                if self.particles[internal_id]['lost_frames'] > self.max_lost:
                    ids_to_delete.append(internal_id)

        for internal_id in ids_to_delete:
            del self.particles[internal_id]

        return result_boxes


# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("✨ Physics Video Tracker Pro")
st.divider()

# Боковая панель
with st.sidebar:
    st.header("📁 Исходные данные")
    uploaded_file = st.file_uploader("Загрузите видео (avi, mp4)", type=['avi', 'mp4'], max_upload_size=1024)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if ret:
        frame_h, frame_w = first_frame.shape[:2]

        col_settings, col_video = st.columns([2, 1], gap="large")

        with col_settings:
            st.subheader("⚙️ Настройки и ROI")
            st.markdown("**Растяните зеленую рамку на изображении:**")

            pil_img = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            box = st_cropper(pil_img, realtime_update=True, box_color='#00FF00', aspect_ratio=None, return_type='box')

            roi_x = box['left']
            roi_y = box['top']
            roi_w = box['width']
            roi_h = box['height']

            with st.expander("⚡ Оптимизация скорости экрана", expanded=False):
                render_mode = st.select_slider(
                    "Частота обновления интерфейса",
                    options=["Каждый кадр", "Х2 (быстро)", "Х5 (очень быстро)", "Только результат (Турбо)"],
                    value="Х2 (быстро)"
                )

            st.divider()
            start_btn = st.button("🚀 ЗАПУСТИТЬ ТРЕКИНГ", type="primary", use_container_width=True)

        # ПРАВАЯ КОЛОНКА ПУСТАЯ ПОКА НЕ НАЖАТА КНОПКА
        if start_btn:
            with col_video:
                st.subheader("📺 Превью / Трансляция")
                video_placeholder = st.empty()

                st.divider()
                st.markdown("### 📊 Аналитика в реальном времени")
                m1, m2, m3 = st.columns(3)
                metric_frame = m1.empty()
                metric_objs = m2.empty()
                metric_area = m3.empty()
                progress_bar = st.progress(0)

            if render_mode == "Каждый кадр":
                render_step = 1
            elif render_mode == "Х2 (быстро)":
                render_step = 2
            elif render_mode == "Х5 (очень быстро)":
                render_step = 5
            else:
                render_step = total_frames

            tracker = PhysicsTracker()
            results_data = []
            frame_count = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # --- ПОДГОТОВКА ПАПОК ДЛЯ ZIP АРХИВА ---
            temp_dir = tempfile.mkdtemp()
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            out_video_path = os.path.join(temp_dir, "tracked_video.mp4")
            out_csv_path = os.path.join(temp_dir, "results.csv")
            zip_file_path = os.path.join(tempfile.gettempdir(), "physics_results.zip")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or math.isnan(fps): fps = 30.0

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_w, frame_h))

            while True:
                ret, frame = cap.read()
                if not ret: break

                frame_count += 1
                roi_img = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

                if roi_img.shape[0] == 0 or roi_img.shape[1] == 0: continue

                hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([35, 40, 90]), np.array([85, 255, 255]))
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detections = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 60:
                        x, y, w, h = cv2.boundingRect(contour)
                        if 0.4 < float(w) / h < 2.5:
                            detections.append(
                                [x + roi_x, y + roi_y, w, h, x + w // 2 + roi_x, y + h // 2 + roi_y, area])

                confirmed_objects = tracker.update(detections, frame_count)

                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
                for obj in confirmed_objects:
                    x, y, w, h, obj_id, cx, cy, area = obj
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    results_data.append({"Frame": frame_count, "ID": obj_id, "X": cx, "Y": cy, "Area": round(area, 2)})

                # Сохраняем видео и отдельные кадры
                out_video.write(frame)
                frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)

                if frame_count % render_step == 0 or frame_count == total_frames:
                    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    metric_frame.metric("Обработано кадров", f"{frame_count} / {total_frames}")
                    metric_objs.metric("Объектов в кадре", len(confirmed_objects))
                    metric_area.metric("Средняя площадь", int(sum([o[7] for o in confirmed_objects]) / len(
                        confirmed_objects)) if confirmed_objects else 0)
                    if total_frames > 0: progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out_video.release()

            # Сохраняем CSV
            pd.DataFrame(results_data).to_csv(out_csv_path, index=False)

            # --- УПАКОВКА В ZIP ---
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(out_video_path, arcname="tracked_video.mp4")
                zipf.write(out_csv_path, arcname="results_data.csv")
                for root, _, files in os.walk(frames_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join("frames", file)
                        zipf.write(file_path, arcname=arcname)

            # Чистим временную папку (оставляем только zip)
            shutil.rmtree(temp_dir)

            st.success(f"✅ Анализ завершен! Создан ZIP-архив с видео, таблицей и {frame_count} кадрами.")

            # Кнопка скачивания архива
            with open(zip_file_path, 'rb') as f:
                st.download_button(
                    label="📦 СКАЧАТЬ ZIP АРХИВ (Видео + Таблица + Кадры)",
                    data=f,
                    file_name="physics_results.zip",
                    mime="application/zip",
                    use_container_width=True
                )