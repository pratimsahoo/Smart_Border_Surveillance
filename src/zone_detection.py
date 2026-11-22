import cv2
import numpy as np

def make_zone(frame_width, frame_height, x1=0.62, y1=0.18, x2=0.96, y2=0.88):
    # Sidebar controls use normalized (0-1) values.
    pts = [
        (int(x1 * frame_width), int(y1 * frame_height)),
        (int(x2 * frame_width), int(y1 * frame_height)),
        (int(x2 * frame_width), int(y2 * frame_height)),
        (int(x1 * frame_width), int(y2 * frame_height)),
    ]
    return np.array(pts, dtype=np.int32)

def point_in_zone(point, polygon):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

def draw_zone(frame, polygon, color=(0, 0, 255)):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
    cv2.putText(frame, "RESTRICTED ZONE", tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2, cv2.LINE_AA)
    return frame
