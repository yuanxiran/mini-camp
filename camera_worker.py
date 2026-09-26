"""Headless camera worker with a small, versioned JSONL protocol.

The worker uses one MediaPipe Gesture Recognizer inference per frame. Its
stdout is reserved for JSONL messages; human-readable diagnostics go to
stderr. Use ``--preview`` only for local visual debugging.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from gesture_demo import GAME_GESTURE_NAMES, top_gesture
from hand_demo import SILENCE_BANNER_MS, MotionTracker, WaveDetector, draw_hand, palm_center


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "gesture_recognizer.task"
PALM_EMIT_INTERVAL_SECONDS = 0.1
SILENCE_STATE_EMIT_INTERVAL_SECONDS = 0.1
CAMERA_READ_FAILURE_THRESHOLD = 3
CAMERA_READ_RETRY_SLEEP_SECONDS = 0.01


def emit(message: dict) -> None:
    """Write exactly one compact JSON object and flush immediately."""
    print(json.dumps(message, separators=(",", ":"), ensure_ascii=True), flush=True)


def timestamp_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def protocol_state(state: str) -> str:
    """Map the existing UI label to the JSONL protocol spelling."""
    return "NO_HAND" if state == "NO HAND" else state


def clamp_unit(value: float) -> float:
    """Clamp a normalized coordinate only at the external PALM boundary."""
    return min(max(float(value), 0.0), 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe hand camera JSONL worker")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show an OpenCV debug window; stdout remains JSONL-only",
    )
    return parser.parse_args()


def preview_frame(frame, result, snapshot, official_gesture, confidence, fps, cv2) -> None:
    """Draw optional local diagnostics without writing anything to stdout."""
    if result.hand_landmarks:
        draw_hand(frame, result.hand_landmarks[0], cv2)

    game_gesture = GAME_GESTURE_NAMES.get(official_gesture, "NONE")
    lines = [
        f"FPS: {fps:5.1f}",
        f"Official gesture: {official_gesture}",
        f"Confidence: {confidence:.2f}",
        f"Game gesture: {game_gesture}",
        f"Motion: {snapshot.motion:.3f}",
        f"State: {snapshot.state}",
        f"still_ms: {snapshot.still_ms}",
        "Press Q to quit",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (12, 28 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Camera Worker Preview", frame)


def run() -> int:
    args = parse_args()
    started = time.perf_counter()

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 1

    if not MODEL_PATH.is_file():
        print(f"Model file not found: {MODEL_PATH}", file=sys.stderr)
        return 1

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    camera = None
    recognizer = None
    try:
        try:
            recognizer = vision.GestureRecognizer.create_from_options(options)
        except Exception as exc:
            emit({"v": 1, "type": "CAMERA_STATE", "available": False, "ts_ms": timestamp_ms(started)})
            print(f"Could not load Gesture Recognizer: {exc}", file=sys.stderr)
            return 1
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            emit({"v": 1, "type": "CAMERA_STATE", "available": False, "ts_ms": timestamp_ms(started)})
            print("Could not open camera index 0.", file=sys.stderr)
            return 1

        emit({"v": 1, "type": "CAMERA_STATE", "available": True, "ts_ms": timestamp_ms(started)})

        motion_tracker = MotionTracker()
        wave_detector = WaveDetector()
        previous_wave_active = False
        silence_banner_until = 0.0
        last_gesture = None
        last_palm_emit_at = None
        last_silence_emit_at = None
        last_silence_state = None
        consecutive_read_failures = 0
        stream_available = True
        tracking_gap = False
        frame_count = 0
        fps = 0.0
        fps_started = time.perf_counter()
        last_timestamp_ms = -1

        while True:
            ok, frame = camera.read()
            if not ok:
                consecutive_read_failures += 1
                tracking_gap = True
                if (
                    consecutive_read_failures >= CAMERA_READ_FAILURE_THRESHOLD
                    and stream_available
                ):
                    emit({"v": 1, "type": "CAMERA_STATE", "available": False, "ts_ms": timestamp_ms(started)})
                    print(
                        f"Camera read failed {consecutive_read_failures} consecutive times.",
                        file=sys.stderr,
                    )
                    stream_available = False
                    # Drop frame-derived state. A later successful frame must
                    # establish fresh baselines and cannot reuse stale data.
                    motion_tracker.no_hand()
                    wave_detector.reset_tracking()
                    previous_wave_active = False
                    last_gesture = None
                    last_palm_emit_at = None
                    last_silence_emit_at = None
                    last_silence_state = None
                time.sleep(CAMERA_READ_RETRY_SLEEP_SECONDS)
                continue

            if not stream_available:
                emit({"v": 1, "type": "CAMERA_STATE", "available": True, "ts_ms": timestamp_ms(started)})
                stream_available = True
            consecutive_read_failures = 0

            if tracking_gap:
                # Even a short read gap must not become a synthetic movement
                # or stillness interval when the next frame arrives.
                motion_tracker.no_hand()
                wave_detector.reset_tracking()
                previous_wave_active = False
                last_palm_emit_at = None
                last_silence_emit_at = None
                last_silence_state = None
                tracking_gap = False

            now = time.perf_counter()
            current_ts_ms = timestamp_ms(started)
            frame_count += 1
            elapsed = now - fps_started
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                fps_started = now

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            mp_timestamp_ms = time.monotonic_ns() // 1_000_000
            if mp_timestamp_ms <= last_timestamp_ms:
                mp_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = mp_timestamp_ms
            result = recognizer.recognize_for_video(mp_image, mp_timestamp_ms)

            official_gesture, confidence = top_gesture(result)
            game_gesture = GAME_GESTURE_NAMES.get(official_gesture, "NONE")
            if not result.hand_landmarks:
                official_gesture = "None"
                confidence = 0.0
                game_gesture = "NONE"

            if game_gesture != last_gesture:
                emit({
                    "v": 1,
                    "type": "GESTURE",
                    "gesture": game_gesture,
                    "confidence": confidence,
                    "ts_ms": current_ts_ms,
                })
                last_gesture = game_gesture

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                palm_x, palm_y = palm_center(landmarks)
                wave_detector.update(now, palm_x)
                wave_active = wave_detector.is_wave(now)
                snapshot = motion_tracker.update(now, landmarks, force_moving=wave_active)

                if last_palm_emit_at is None or now - last_palm_emit_at >= PALM_EMIT_INTERVAL_SECONDS:
                    emit({
                        "v": 1,
                        "type": "PALM",
                        "x": clamp_unit(palm_x),
                        "y": clamp_unit(palm_y),
                        "ts_ms": current_ts_ms,
                    })
                    last_palm_emit_at = now

                if wave_active and not previous_wave_active:
                    emit({"v": 1, "type": "WAVE", "ts_ms": current_ts_ms})
                previous_wave_active = wave_active

                if snapshot.silence_reached:
                    emit({"v": 1, "type": "SILENCE_REACHED", "ts_ms": current_ts_ms})
                    silence_banner_until = now + SILENCE_BANNER_MS / 1000.0
            else:
                snapshot = motion_tracker.no_hand()
                wave_detector.reset_tracking()
                previous_wave_active = False
                last_palm_emit_at = None
                silence_banner_until = 0.0

            json_state = protocol_state(snapshot.state)
            if (
                last_silence_emit_at is None
                or json_state != last_silence_state
                or now - last_silence_emit_at >= SILENCE_STATE_EMIT_INTERVAL_SECONDS
            ):
                emit({
                    "v": 1,
                    "type": "SILENCE_STATE",
                    "motion": snapshot.motion,
                    "still_ms": snapshot.still_ms,
                    "state": json_state,
                    "ts_ms": current_ts_ms,
                })
                last_silence_emit_at = now
                last_silence_state = json_state

            if args.preview:
                preview_frame(frame, result, snapshot, official_gesture, confidence, fps, cv2)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                try:
                    if cv2.getWindowProperty("Camera Worker Preview", cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

    except KeyboardInterrupt:
        print("Worker interrupted.", file=sys.stderr)
    except Exception as exc:  # Keep stdout clean even for unexpected runtime errors.
        print(f"Worker error: {exc}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            camera.release()
        if args.preview:
            cv2.destroyAllWindows()
        if recognizer is not None:
            recognizer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
