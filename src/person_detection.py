from ultralytics import YOLO

def load_detector(model_name="yolov8n.pt"):
    # Ultralytics downloads pretrained weights on first use if not cached locally.
    return YOLO(model_name)

def detect_and_track(model, frame, confidence=0.35, image_size=640):
    # COCO class 0 = person. persist=True carries tracker state across frames.
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )
    return results[0]
