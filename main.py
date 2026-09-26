"""Minimal local MediaPipe face-action demo.

The application reads frames from the default local camera, runs MediaPipe
Face Landmarker blendshape inference, and displays the results in an OpenCV
window. No frames or face data are saved or uploaded.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "face_landmarker.task"

# Blendshape thresholds. A small hysteresis gap prevents rapid state flipping.
ENTER_THRESHOLD = 0.55
EXIT_THRESHOLD = 0.45
STABLE_FRAMES = 3


def classify_action(blendshapes: dict[str, float]) -> str:
    """Return the highest-priority stable candidate action for one frame."""
    left_blink = blendshapes.get("eyeBlinkLeft", 0.0)
    right_blink = blendshapes.get("eyeBlinkRight", 0.0)
    jaw_open = blendshapes.get("jawOpen", 0.0)
    brow_up = blendshapes.get("browInnerUp", 0.0)
    smile = (
        blendshapes.get("mouthSmileLeft", 0.0)
        + blendshapes.get("mouthSmileRight", 0.0)
    ) / 2.0

    # The priority avoids reporting a single-eye state while both eyes are closed.
    if left_blink >= ENTER_THRESHOLD and right_blink >= ENTER_THRESHOLD:
        return "Both eyes closed"
    if left_blink >= ENTER_THRESHOLD and right_blink <= EXIT_THRESHOLD:
        return "Left eye closed"
    if right_blink >= ENTER_THRESHOLD and left_blink <= EXIT_THRESHOLD:
        return "Right eye closed"
    if jaw_open >= ENTER_THRESHOLD:
        return "Mouth open"
    if smile >= ENTER_THRESHOLD:
        return "Smile"
    if brow_up >= ENTER_THRESHOLD:
        return "Brow raised"
    return "No action"


class StableAction:
    """Apply a short consecutive-frame filter to the displayed action."""

    def __init__(self) -> None:
        self.current = "No action"
        self.candidate = self.current
        self.count = 0

    def update(self, action: str) -> str:
        if action == self.candidate:
            self.count += 1
        else:
            self.candidate = action
            self.count = 1
        if self.count >= STABLE_FRAMES:
            self.current = self.candidate
        return self.current


def _blendshape_values(result) -> dict[str, float]:
    """Extract the first face's blendshape scores into a simple dictionary."""
    if not result.face_blendshapes:
        return {}
    return {category.category_name: category.score for category in result.face_blendshapes[0]}


def run() -> int:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        print("Activate .venv and install requirements.txt first.", file=sys.stderr)
        return 1

    if not MODEL_PATH.is_file():
        print(f"Model file not found: {MODEL_PATH}", file=sys.stderr)
        print("Download the official Face Landmarker model as described in README.md.", file=sys.stderr)
        return 1

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        print("Could not open camera index 0.", file=sys.stderr)
        print("Check Windows camera permission and close other camera applications.", file=sys.stderr)
        return 1

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    smoother = StableAction()
    frame_count = 0
    fps = 0.0
    fps_started = time.perf_counter()
    last_timestamp_ms = -1

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read a frame from camera.", file=sys.stderr)
                    break

                frame_count += 1
                now = time.perf_counter()
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
                values = _blendshape_values(result)
                action = smoother.update(classify_action(values))

                lines = [
                    f"FPS: {fps:5.1f}",
                    f"Action: {action}",
                    f"eyeBlinkLeft:  {values.get('eyeBlinkLeft', 0.0):.2f}",
                    f"eyeBlinkRight: {values.get('eyeBlinkRight', 0.0):.2f}",
                    f"jawOpen:       {values.get('jawOpen', 0.0):.2f}",
                    f"browInnerUp:   {values.get('browInnerUp', 0.0):.2f}",
                    f"mouthSmileLeft:{values.get('mouthSmileLeft', 0.0):.2f}",
                    f"mouthSmileRight:{values.get('mouthSmileRight', 0.0):.2f}",
                    "Press Q to quit",
                ]
                for index, text in enumerate(lines):
                    cv2.putText(
                        frame,
                        text,
                        (12, 28 + index * 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("MediaPipe Face Actions", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
