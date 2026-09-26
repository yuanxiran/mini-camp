"""Minimal local MediaPipe hand, palm, and wave prototype.

This demo keeps all camera processing local. It draws one detected hand,
reports a normalized palm center, uses the recent palm-x trajectory for a
small WAVE candidate detector, and reports a one-shot SILENCE_REACHED event
after valid continuous stillness. It intentionally does not implement JSONL
output or any computer-control behavior.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from math import hypot
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
WINDOW_NAME = "MediaPipe Hand Demo"

# Motion and stillness defaults. These are deliberately simple first-test
# values and are expected to be tuned with real camera observations.
PALM_LANDMARK_INDICES = (0, 5, 9, 13, 17)
MOTION_SMOOTHING_SAMPLES = 5
MIN_VALID_DT_SECONDS = 0.005
MAX_VALID_DT_SECONDS = 0.25
STILL_ENTER_THRESHOLD = 0.08
MOVING_RESET_THRESHOLD = 0.15
SILENCE_DURATION_MS = 2000
SILENCE_BANNER_MS = 800

# WAVE prototype configuration.
WAVE_WINDOW_SECONDS = 1.2
WAVE_DIRECTION_DELTA = 0.025
WAVE_MIN_SAMPLES = 8
WAVE_MIN_SPAN = 0.12
WAVE_MIN_PATH = 0.28
WAVE_MIN_REVERSALS = 2
WAVE_DISPLAY_SECONDS = 0.6

# A hand skeleton has 21 landmarks. These are the standard MediaPipe indices.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class MotionSnapshot:
    """Reusable internal output for a future SILENCE_STATE wrapper."""

    palm_x: float
    palm_y: float
    horizontal_motion: float
    motion: float
    still_ms: int
    state: str
    silence_triggered: bool
    silence_reached: bool


class MotionTracker:
    """Track smoothed hand motion and a one-shot continuous-stillness event."""

    def __init__(self) -> None:
        self.previous_points: tuple[tuple[float, float], ...] | None = None
        self.previous_time: float | None = None
        self.motion_samples: deque[float] = deque(maxlen=MOTION_SMOOTHING_SAMPLES)
        self.state = "MOVING"
        self.still_ms = 0.0
        self.silence_triggered = False

    @staticmethod
    def _points(landmarks) -> tuple[tuple[float, float], ...]:
        return tuple(
            (landmarks[index].x, landmarks[index].y)
            for index in PALM_LANDMARK_INDICES
        )

    @staticmethod
    def _center(points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    def _baseline_snapshot(
        self,
        now: float,
        points: tuple[tuple[float, float], ...],
    ) -> MotionSnapshot:
        self.previous_points = points
        self.previous_time = now
        self.motion_samples.clear()
        self.state = "MOVING"
        self.still_ms = 0.0
        palm_x, palm_y = self._center(points)
        return MotionSnapshot(
            palm_x=palm_x,
            palm_y=palm_y,
            horizontal_motion=0.0,
            motion=0.0,
            still_ms=0,
            state=self.state,
            silence_triggered=self.silence_triggered,
            silence_reached=False,
        )

    def update(self, now: float, landmarks, force_moving: bool = False) -> MotionSnapshot:
        points = self._points(landmarks)
        if self.previous_points is None or self.previous_time is None:
            return self._baseline_snapshot(now, points)

        dt = now - self.previous_time
        if not MIN_VALID_DT_SECONDS <= dt <= MAX_VALID_DT_SECONDS:
            return self._baseline_snapshot(now, points)

        previous_points = self.previous_points
        previous_palm_x, _ = self._center(previous_points)
        palm_x, palm_y = self._center(points)
        point_speeds = (
            hypot(current[0] - previous[0], current[1] - previous[1]) / dt
            for current, previous in zip(points, previous_points)
        )
        raw_motion = sum(point_speeds) / len(points)
        horizontal_motion = (palm_x - previous_palm_x) / dt
        self.previous_points = points
        self.previous_time = now
        self.motion_samples.append(raw_motion)
        motion = sum(self.motion_samples) / len(self.motion_samples)

        if force_moving or motion >= MOVING_RESET_THRESHOLD:
            self.state = "MOVING"
            self.still_ms = 0.0
            self.silence_triggered = False
        elif motion <= STILL_ENTER_THRESHOLD:
            self.state = "STILL"
        # In the hysteresis band, retain the previous MOVING/STILL state.

        silence_reached = False
        if self.state == "STILL":
            previous_still_ms = self.still_ms
            self.still_ms += dt * 1000.0
            if (
                not self.silence_triggered
                and previous_still_ms < SILENCE_DURATION_MS <= self.still_ms
            ):
                self.silence_triggered = True
                silence_reached = True
        else:
            self.still_ms = 0.0

        return MotionSnapshot(
            palm_x=palm_x,
            palm_y=palm_y,
            horizontal_motion=horizontal_motion,
            motion=motion,
            still_ms=int(self.still_ms),
            state=self.state,
            silence_triggered=self.silence_triggered,
            silence_reached=silence_reached,
        )

    def no_hand(self) -> MotionSnapshot:
        """Reset tracking continuity but preserve the one-shot event lock."""
        self.previous_points = None
        self.previous_time = None
        self.motion_samples.clear()
        self.state = "NO HAND"
        self.still_ms = 0.0
        return MotionSnapshot(
            palm_x=0.0,
            palm_y=0.0,
            horizontal_motion=0.0,
            motion=0.0,
            still_ms=0,
            state=self.state,
            silence_triggered=self.silence_triggered,
            silence_reached=False,
        )


class WaveDetector:
    """Detect clear left-right oscillation from a short palm-x history."""

    def __init__(self, window_seconds: float = WAVE_WINDOW_SECONDS) -> None:
        self.samples: deque[tuple[float, float]] = deque()
        self.window_seconds = window_seconds
        self.last_direction = 0
        self.reversals = 0
        self.wave_until = 0.0
        self.last_x: float | None = None
        self.last_time: float | None = None

    def reset_tracking(self) -> None:
        self.samples.clear()
        self.last_direction = 0
        self.reversals = 0
        self.wave_until = 0.0
        self.last_x = None
        self.last_time = None

    def update(self, now: float, palm_x: float) -> None:
        if self.last_x is None or self.last_time is None:
            self.samples.append((now, palm_x))
            self.last_x = palm_x
            self.last_time = now
            return

        dt = now - self.last_time
        if not MIN_VALID_DT_SECONDS <= dt <= MAX_VALID_DT_SECONDS:
            self.reset_tracking()
            self.samples.append((now, palm_x))
            self.last_x = palm_x
            self.last_time = now
            return

        delta = palm_x - self.last_x
        self.last_x = palm_x
        self.last_time = now

        self.samples.append((now, palm_x))
        while self.samples and now - self.samples[0][0] > self.window_seconds:
            self.samples.popleft()

        direction = (
            1 if delta > WAVE_DIRECTION_DELTA
            else -1 if delta < -WAVE_DIRECTION_DELTA
            else 0
        )
        if direction and self.last_direction and direction != self.last_direction:
            self.reversals += 1
        if direction:
            self.last_direction = direction

        xs = [sample_x for _, sample_x in self.samples]
        span = max(xs) - min(xs) if xs else 0.0
        path = sum(abs(xs[index] - xs[index - 1]) for index in range(1, len(xs)))
        if (
            len(xs) >= WAVE_MIN_SAMPLES
            and span >= WAVE_MIN_SPAN
            and path >= WAVE_MIN_PATH
            and self.reversals >= WAVE_MIN_REVERSALS
        ):
            self.wave_until = now + WAVE_DISPLAY_SECONDS
            self.reversals = 0

    def is_wave(self, now: float) -> bool:
        return now < self.wave_until


def palm_center(landmarks) -> tuple[float, float]:
    """Average wrist and four MCP landmarks for a stable palm center."""
    return (
        sum(landmarks[index].x for index in PALM_LANDMARK_INDICES)
        / len(PALM_LANDMARK_INDICES),
        sum(landmarks[index].y for index in PALM_LANDMARK_INDICES)
        / len(PALM_LANDMARK_INDICES),
    )


def draw_hand(frame, landmarks, cv2) -> None:
    height, width = frame.shape[:2]
    points = [
        (max(0, min(width - 1, int(point.x * width))),
         max(0, min(height - 1, int(point.y * height))))
        for point in landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 180, 0), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1, cv2.LINE_AA)


def run() -> int:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        print("Use the project's .venv and install requirements.txt.", file=sys.stderr)
        return 1

    if not MODEL_PATH.is_file():
        print(f"Model file not found: {MODEL_PATH}", file=sys.stderr)
        return 1

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        print("Could not open camera index 0.", file=sys.stderr)
        return 1

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    wave_detector = WaveDetector()
    motion_tracker = MotionTracker()
    silence_banner_until = 0.0
    frame_count = 0
    fps = 0.0
    fps_started = time.perf_counter()
    last_timestamp_ms = -1

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read a frame from camera.", file=sys.stderr)
                    break

                now = time.perf_counter()
                frame_count += 1
                elapsed = now - fps_started
                if elapsed >= 0.5:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_started = now

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = time.monotonic_ns() // 1_000_000
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                status = "NO HAND"
                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    draw_hand(frame, landmarks, cv2)
                    palm_x, palm_y = palm_center(landmarks)
                    wave_detector.update(now, palm_x)
                    wave_active = wave_detector.is_wave(now)
                    snapshot = motion_tracker.update(
                        now,
                        landmarks,
                        force_moving=wave_active,
                    )
                    if snapshot.silence_reached:
                        silence_banner_until = now + SILENCE_BANNER_MS / 1000.0
                    height, width = frame.shape[:2]
                    palm_point = (int(palm_x * width), int(palm_y * height))
                    cv2.circle(frame, palm_point, 10, (0, 0, 255), 2, cv2.LINE_AA)
                    status = "WAVE" if wave_active else "TRACKING"
                else:
                    snapshot = motion_tracker.no_hand()
                    wave_detector.reset_tracking()
                    silence_banner_until = 0.0

                lines = [
                    f"FPS: {fps:5.1f}",
                    f"Status: {status}",
                    f"palm_x: {snapshot.palm_x:.3f}",
                    f"palm_y: {snapshot.palm_y:.3f}",
                    f"horizontal_motion: {snapshot.horizontal_motion:+.3f}",
                    f"motion: {snapshot.motion:.3f}",
                    f"still_ms: {snapshot.still_ms}",
                    f"Motion state: {snapshot.state}",
                    f"Silence triggered: {'YES' if snapshot.silence_triggered else 'NO'}",
                    "Press Q to quit",
                ]
                for index, text in enumerate(lines):
                    cv2.putText(
                        frame,
                        text,
                        (12, 28 + index * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.68,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                if now < silence_banner_until:
                    cv2.putText(
                        frame,
                        "SILENCE_REACHED",
                        (12, frame.shape[0] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
