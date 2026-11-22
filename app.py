
from pathlib import Path
import time
import cv2
import streamlit as st
import pandas as pd

from src.video_input import discover_videos, open_video
from src.person_detection import load_detector, detect_and_track
from src.object_tracking import extract_tracks
from src.zone_detection import make_zone, point_in_zone, draw_zone
from src.reliability import TemporalConfirmation
from src.alert_system import save_alert


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Border Surveillance AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Smart Border Surveillance Using AI")
st.caption(
    "AI-based person detection, tracking, restricted-zone monitoring "
    "and intrusion alerting for human review."
)


# =========================================================
# 2. SIDEBAR SETTINGS
# =========================================================

with st.sidebar:
    st.header("⚙️ Prototype Settings")

    conf = st.slider(
        "YOLO Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
    )

    imgsz = st.select_slider(
        "Inference Image Size",
        options=[320, 416, 512, 640, 768],
        value=640,
    )

    required_hits = st.slider(
        "Frames required to confirm intrusion",
        min_value=1,
        max_value=15,
        value=4,
    )

    cooldown = st.slider(
        "Alert cooldown (frames)",
        min_value=0,
        max_value=180,
        value=45,
        step=5,
    )

    st.divider()
    st.subheader("Restricted Zone")

    zone_x1 = st.slider("Left", 0.00, 0.90, 0.62, 0.02)
    zone_y1 = st.slider("Top", 0.00, 0.90, 0.18, 0.02)
    zone_x2 = st.slider("Right", 0.10, 1.00, 0.96, 0.02)
    zone_y2 = st.slider("Bottom", 0.10, 1.00, 0.88, 0.02)

    st.divider()

    run = st.button(
        "▶ Start Processing",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# 3. VIDEO SELECTION
# =========================================================

videos = discover_videos("dataset")

uploaded = st.file_uploader(
    "Upload CCTV video (or select one from your dataset below)",
    type=["mp4", "avi", "mov", "mkv"],
)

video_path = None
video_label = None

if uploaded is not None:

    upload_dir = Path("dataset/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    video_path = upload_dir / uploaded.name
    video_path.write_bytes(uploaded.getbuffer())

    video_label = uploaded.name

elif videos:

    video_options = [str(p) for p in videos]

    selected_video = st.selectbox(
        "Select a dataset video",
        options=video_options,
        format_func=lambda p: f"{Path(p).parent.name} / {Path(p).name}",
    )

    video_path = Path(selected_video)
    video_label = str(video_path)

else:

    st.warning("No videos found inside dataset folder.")


# =========================================================
# 4. TOP METRICS
# =========================================================

m1, m2, m3 = st.columns(3)

m1.metric("Videos Available", len(videos))
m2.metric("Detection Model", "YOLOv8 Nano")
m3.metric("Tracking Algorithm", "ByteTrack")


# =========================================================
# 5. TWO-COLUMN MONITORING LAYOUT
# =========================================================

st.divider()

video_col, monitor_col = st.columns([3.2, 1.15], gap="medium")

with video_col:

    st.subheader("📹 CCTV Monitoring")

    video_area = st.empty()

    video_area.info(
        "Select a video and click Start Processing "
        "to begin surveillance analysis."
    )

with monitor_col:

    st.subheader("📊 Live Monitoring")

    status_area = st.empty()

    frame_metric = st.empty()
    track_metric = st.empty()
    alert_metric = st.empty()
    fps_metric = st.empty()

    progress_area = st.empty()

    alert_area = st.empty()


# =========================================================
# 6. VIDEO PROCESSING
# =========================================================

if run:

    if video_path is None:

        st.error("Please select or upload a video first.")

    elif zone_x2 <= zone_x1 or zone_y2 <= zone_y1:

        st.error("Invalid restricted-zone coordinates.")

    else:

        cap = None

        try:

            # -------------------------------------------------
            # Load YOLO
            # -------------------------------------------------

            status_area.info("Loading YOLO model...")

            model = load_detector("yolov8n.pt")

            # -------------------------------------------------
            # Open video
            # -------------------------------------------------

            cap, meta = open_video(video_path)

            total_frames = meta["frame_count"]
            video_fps = meta["fps"]

            status_area.success("Video opened successfully")

            # -------------------------------------------------
            # Display video information
            # -------------------------------------------------

            with monitor_col:

                st.caption("VIDEO INFORMATION")

                st.write(f"Resolution: {meta['width']} × {meta['height']}")
                st.write(f"Original FPS: {video_fps:.2f}")
                st.write(f"Total frames: {total_frames}")

            # -------------------------------------------------
            # Initialize temporal confirmation
            # -------------------------------------------------

            confirmer = TemporalConfirmation(
                required_hits=required_hits,
                cooldown_frames=cooldown,
            )

            frame_index = 0
            alerts_this_run = 0

            progress_bar = progress_area.progress(0)

            processing_start = time.perf_counter()

            # -------------------------------------------------
            # Read video frames
            # -------------------------------------------------

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frame_index += 1

                h, w = frame.shape[:2]

                # -------------------------------------------------
                # Create restricted zone
                # -------------------------------------------------

                zone = make_zone(
                    w,
                    h,
                    zone_x1,
                    zone_y1,
                    zone_x2,
                    zone_y2,
                )

                # -------------------------------------------------
                # YOLO detection and ByteTrack tracking
                # -------------------------------------------------

                result = detect_and_track(
                    model,
                    frame,
                    confidence=conf,
                    image_size=imgsz,
                )

                tracks = extract_tracks(result)

                # -------------------------------------------------
                # Draw restricted zone
                # -------------------------------------------------

                display_frame = draw_zone(frame.copy(), zone)

                # -------------------------------------------------
                # Process detected people
                # -------------------------------------------------

                intrusion_detected = False

                for track in tracks:

                    tid = track["track_id"]

                    x1, y1, x2, y2 = track["box"]

                    inside = point_in_zone(
                        track["anchor"],
                        zone,
                    )

                    trigger = (
                        confirmer.update(tid, inside, frame_index)
                        if tid is not None
                        else False
                    )

                    # Colors
                    color = (0, 0, 255) if inside else (0, 255, 0)

                    # Bounding box
                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    # Tracking ID and confidence
                    label = (
                        f"ID {tid if tid is not None else '?'} "
                        f"| {track['confidence']:.2f}"
                    )

                    cv2.putText(
                        display_frame,
                        label,
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                    # Bottom-center anchor
                    cv2.circle(
                        display_frame,
                        track["anchor"],
                        5,
                        color,
                        -1,
                    )

                    # -------------------------------------------------
                    # Intrusion alert
                    # -------------------------------------------------

                    if trigger:

                        intrusion_detected = True
                        alerts_this_run += 1

                        cv2.putText(
                            display_frame,
                            "INTRUSION ALERT!",
                            (25, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            3,
                        )

                        # Save image and alert CSV
                        save_alert(
                            display_frame,
                            tid,
                            str(video_label),
                            frame_index,
                            track["confidence"],
                        )

                # -------------------------------------------------
                # Resize video for screen display
                # -------------------------------------------------

                display_max_width = 850
                display_max_height = 520

                scale = min(
                    display_max_width / w,
                    display_max_height / h,
                    1.0,
                )

                if scale < 1.0:

                    new_width = int(w * scale)
                    new_height = int(h * scale)

                    display_frame = cv2.resize(
                        display_frame,
                        (new_width, new_height),
                        interpolation=cv2.INTER_AREA,
                    )

                # -------------------------------------------------
                # Update video on LEFT
                # -------------------------------------------------

                video_area.image(
                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )

                # -------------------------------------------------
                # Update monitoring panel on RIGHT
                # -------------------------------------------------

                elapsed = time.perf_counter() - processing_start

                processing_fps = (
                    frame_index / elapsed if elapsed > 0 else 0
                )

                frame_metric.metric(
                    "Frames Processed",
                    f"{frame_index}/{total_frames if total_frames else '?'}",
                )

                track_metric.metric(
                    "Persons Detected",
                    len(tracks),
                )

                alert_metric.metric(
                    "Alerts Generated",
                    alerts_this_run,
                )

                fps_metric.metric(
                    "Processing FPS",
                    f"{processing_fps:.2f}",
                )

                if intrusion_detected:

                    status_area.error("⚠️ INTRUSION DETECTED")

                    alert_area.warning(
                        f"Confirmed intrusion alert! "
                        f"Total alerts: {alerts_this_run}"
                    )

                else:

                    status_area.success("Monitoring — No confirmed intrusion")

                # -------------------------------------------------
                # Update progress
                # -------------------------------------------------

                if total_frames:

                    progress_bar.progress(
                        min(frame_index / total_frames, 1.0)
                    )

            # -------------------------------------------------
            # End of video
            # -------------------------------------------------

            cap.release()
            cap = None

            progress_area.empty()

            status_area.info("Video processing completed")

            st.success(
                f"Processing complete! "
                f"Frames: {frame_index} | "
                f"Alerts: {alerts_this_run}"
            )

        except Exception as exc:

            if cap is not None:
                cap.release()

            st.error("An error occurred during video processing.")
            st.exception(exc)


# =========================================================
# 7. ALERT LOG (COLLAPSED TO SAVE SCREEN SPACE)
# =========================================================

with st.expander("📋 View Alert History", expanded=False):

    csv_path = Path("alerts/alert_logs/intrusion_alerts.csv")

    if csv_path.exists():

        try:

            df = pd.read_csv(csv_path)

            st.dataframe(
                df.tail(100),
                use_container_width=True,
                height=300,
            )

            st.download_button(
                "⬇ Download Alert CSV",
                data=df.to_csv(index=False),
                file_name="intrusion_alerts.csv",
                mime="text/csv",
            )

        except Exception as e:

            st.error(f"Could not read alert log: {e}")

    else:

        st.info("No alerts generated yet.")


# =========================================================
# 8. FOOTER
# =========================================================

st.caption(
    "Research prototype only. Alerts are intended for human verification. "
    "Current zone logic uses a rectangle and temporal frame-count confirmation."
)
