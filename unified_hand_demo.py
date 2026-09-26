"""Unified hand validation demo backed only by Gesture Recognizer.

One Gesture Recognizer inference supplies both official canned gestures and
hand landmarks. The existing palm, motion, wave, and stillness logic then uses
those landmarks without running a separate Hand Landmarker.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from gesture_demo import GAME_GESTURE_NAMES, top_gesture
from hand_demo import (
    SILENCE_BANNER_MS,
    MotionTracker,
    WaveDetector,
    draw_hand,
    palm_center,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "gesture_recognizer.task"
WINDOW_NAME = "Unified MediaPipe Hand Demo"


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
    motion_tracker = MotionTracker()
    wave_detector = WaveDetector()
    silence_banner_until = 0.0
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

                official_gesture, confidence = top_gesture(result)
                game_gesture = GAME_GESTURE_NAMES.get(official_gesture, "NONE")
                status = "NO HAND"

                if result.hand_landmarks:
                    # These landmarks come from this same Gesture Recognizer result.
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
                    cv2.circle(
                        frame,
                        (int(palm_x * width), int(palm_y * height)),
                        10,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    status = "WAVE" if wave_active else "TRACKING"
                else:
                    # Preserve the prior bug fix: reset only when no hand exists.
                    snapshot = motion_tracker.no_hand()
                    wave_detector.reset_tracking()
                    silence_banner_until = 0.0

                lines = [
                    f"FPS: {fps:5.1f}",
                    f"Status: {status}",
                    f"Official gesture: {official_gesture}",
                    f"Confidence: {confidence:.2f}",
                    f"Game gesture: {game_gesture}",
                    f"Hands: {len(result.hand_landmarks)}",
                    f"palm_x: {snapshot.palm_x:.3f}",
                    f"palm_y: {snapshot.palm_y:.3f}",
                    f"horizontal_motion: {snapshot.horizontal_motion:+.3f}",
                    f"motion: {snapshot.motion:.3f}",
                    f"Motion state: {snapshot.state}",
                    f"still_ms: {snapshot.still_ms}",
                    f"Silence triggered: {'YES' if snapshot.silence_triggered else 'NO'}",
                    f"WAVE: {'YES' if status == 'WAVE' else 'NO'}",
                    "Press Q to quit",
                ]
                for index, text in enumerate(lines):
                    cv2.putText(
                        frame,
                        text,
                        (12, 26 + index * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
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
