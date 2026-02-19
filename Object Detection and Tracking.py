import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tempfile
import os

# Force CPU (for Windows stability)
device = "cpu"

# Load YOLO model once
model = YOLO("yolov8n.pt")
model.to(device)


def process_video(video_file):

    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(video_file)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps > 60:
        fps = 25  # stable fallback fps

    # Save output video in temp folder
    output_path = os.path.join(tempfile.gettempdir(), "tracked_output.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    object_ids = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO detection
        results = model(frame, device=device)

        detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box
                detections.append(([x1, y1, x2 - x1, y2 - y1], float(conf), int(cls)))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            object_ids.add(track_id)

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame,
                        f'ID: {track_id}',
                        (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

        # Total count text
        cv2.putText(frame,
                    f'Total Objects: {len(object_ids)}',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

        out.write(frame)

    cap.release()
    out.release()

    # Create summary statement
    summary = f"""
    ✅ Video Processing Complete!

    🎥 Total Frames Processed: {frame_count}
    👥 Total Unique Objects Detected: {len(object_ids)}

    Tracking powered by YOLOv8 + DeepSORT.
    """

    return output_path, summary


# Gradio Interface
interface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.Video(label="Processed Video"),
        gr.Textbox(label="Processing Summary")
    ],
    title="VisionTrack AI - Object Detection & Tracking",
    description="Upload a video to detect and track objects using YOLOv8 + DeepSORT."
)

if __name__ == "__main__":
    interface.launch()
