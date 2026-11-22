from pathlib import Path
from datetime import datetime
import csv
import cv2

EVIDENCE_DIR = Path("alerts/evidence")
LOG_DIR = Path("alerts/alert_logs")
CSV_PATH = LOG_DIR / "intrusion_alerts.csv"

def save_alert(frame, track_id, video_name, frame_index, confidence):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_video = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(video_name).stem)
    image_name = f"{safe_video}_track{track_id}_frame{frame_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image_path = EVIDENCE_DIR / image_name
    cv2.imwrite(str(image_path), frame)

    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "video", "track_id", "frame_index", "confidence", "evidence_path"])
        writer.writerow([timestamp, video_name, track_id, frame_index, f"{confidence:.3f}", str(image_path)])
    return {"timestamp": timestamp, "image_path": str(image_path), "csv_path": str(CSV_PATH)}
