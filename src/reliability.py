from collections import defaultdict

class TemporalConfirmation:
    """Baseline temporal rule: alert after a track is in-zone N processed frames."""
    def __init__(self, required_hits=4, cooldown_frames=45):
        self.required_hits = max(1, int(required_hits))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.zone_hits = defaultdict(int)
        self.last_alert_frame = {}
        self.alerted_tracks = set()

    def update(self, track_id, is_inside, frame_index):
        if track_id is None:
            return False
        if is_inside:
            self.zone_hits[track_id] += 1
        else:
            self.zone_hits[track_id] = 0
            self.alerted_tracks.discard(track_id)

        if not is_inside or self.zone_hits[track_id] < self.required_hits:
            return False

        # Avoid repeated alerts while the same track remains inside the zone.
        if track_id in self.alerted_tracks:
            return False

        last = self.last_alert_frame.get(track_id, -10**9)
        if frame_index - last < self.cooldown_frames:
            return False
        self.last_alert_frame[track_id] = frame_index
        self.alerted_tracks.add(track_id)
        return True
