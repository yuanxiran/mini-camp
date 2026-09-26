"""Standalone MediaPipe Gesture Recognizer validation demo.

This demo uses the official Gesture Recognizer task model to display the
canned gesture category and score together with the hand landmarks returned by
the same inference call. Camera frames and results remain in local memory.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "gesture_recognizer.task"
WINDOW_NAME = "MediaPipe Gesture Recognizer Demo"

# Standard MediaPipe hand landmark connections.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

GAME_GESTURE_NAMES = {
    "Open_Palm": "PALM",
    "Thumb_Up": "GOOD",
}


def draw_hand(frame, landmarks, cv2) -> None:
    height, width = frame.shape[:2]
    points = [
        (
            max(0, min(width - 1, int(point.x * width))),
            max(0, min(height - 1, int(point.y * height))),
        )
        for point in landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 180, 0), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1, cv2.LINE_AA)


def top_gesture(result) -> tuple[str, float]:
    """Read the official top category for the first detected hand."""
    if not result.gestures or not result.gestures[0]:
        return "None", 0.0
    category = result.gestures[0][0]
    return category.category_name or "None", float(category.score or 0.0)


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

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        print("Could not open camera index 0.", file=sys.stderr)
        return 1

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    frame_count = 0
    fps = 0.0
    fps_started = time.perf_counter()
    last_timestamp_ms = -1

    try:
        with vision.GestureRecognizer.create_from_options(options) as recognizer:
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
                result = recognizer.recognize_for_video(mp_image, timestamp_ms)

                gesture_name, confidence = top_gesture(result)
                game_gesture = GAME_GESTURE_NAMES.get(gesture_name, "NONE")
                if result.hand_landmarks:
                    draw_hand(frame, result.hand_landmarks[0], cv2)

                lines = [
                    f"Gesture: {gesture_name}",
                    f"Confidence: {confidence:.2f}",
                    f"Game gesture: {game_gesture}",
                    f"Hands: {len(result.hand_landmarks)}",
                    f"FPS: {fps:5.1f}",
                    "Press Q to quit",
                ]
                for index, text in enumerate(lines):
                    cv2.putText(
                        frame,
                        text,
                        (12, 30 + index * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
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
