def extract_tracks(result):
    tracks = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return tracks

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    for i, box in enumerate(xyxy):
        track_id = int(ids[i]) if ids is not None else None
        x1, y1, x2, y2 = map(int, box.tolist())
        # Use bottom-center as a more practical zone-entry anchor than bbox center.
        anchor = ((x1 + x2) // 2, y2)
        tracks.append({
            "track_id": track_id,
            "box": (x1, y1, x2, y2),
            "confidence": float(confs[i]),
            "anchor": anchor,
        })
    return tracks
