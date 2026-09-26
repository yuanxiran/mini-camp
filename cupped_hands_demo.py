"""Standalone two-hand ``CUPPED_HANDS`` pose validation demo.

The demo intentionally stays separate from the JSONL worker and the existing
one-hand demos.  One Gesture Recognizer inference supplies both hands,
landmarks, official gesture labels, and handedness; the pose itself is a small
screen-space geometry rule with real-time enter/exit hold intervals.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from math import hypot
from pathlib import Path

from gesture_demo import draw_hand


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "gesture_recognizer.task"
WINDOW_NAME = "Cupped Hands Demo"

# First-test values. Tune these with a real camera; they are intentionally not
# presented as final calibrated thresholds.
PALM_LANDMARK_INDICES = (0, 5, 9, 13, 17)
MIN_PALM_DISTANCE = 0.12
MAX_PALM_DISTANCE = 0.65
MIN_HORIZONTAL_GAP = 0.08
MAX_PALM_Y_DIFF = 0.20
MIN_OPEN_FINGERS = 3
OPEN_FINGER_DISTANCE_RATIO = 1.08
MIN_INDEX_WRIST_RATIO = 1.20
CUPPED_ENTER_HOLD_SECONDS = 0.40
CUPPED_EXIT_HOLD_SECONDS = 0.15
MIN_WRIST_GAP_X = 1e-6


@dataclass(frozen=True)
class HandInfo:
    landmarks: object
    palm_x: float
    palm_y: float
    official_gesture: str
    confidence: float
    handedness: str


@dataclass(frozen=True)
class PoseChecks:
    two_hands: bool
    distance_ok: bool
    height_ok: bool
    separation_ok: bool
    left_open: bool
    right_open: bool
    spread_shape_ok: bool
    fists_rejected: bool
    wrist_gap_x: float
    index_gap_x: float
    index_wrist_ratio: float

    @property
    def candidate(self) -> bool:
        return (
            self.two_hands
            and self.distance_ok
            and self.height_ok
            and self.separation_ok
            and self.left_open
            and self.right_open
            and self.spread_shape_ok
            and self.fists_rejected
        )


class PoseStabilizer:
    """Enter/leave CUPPED_HANDS after real-time candidate hold intervals."""

    def __init__(self) -> None:
        self.pose = "NONE"
        self.candidate_start_time: float | None = None
        self.failure_start_time: float | None = None
        self.candidate_hold_seconds = 0.0
        self.failure_hold_seconds = 0.0

    def update(self, now: float, candidate: bool) -> str:
        if candidate:
            self.failure_start_time = None
            self.failure_hold_seconds = 0.0
            if self.candidate_start_time is None:
                self.candidate_start_time = now
            self.candidate_hold_seconds = max(0.0, now - self.candidate_start_time)
            if self.pose == "NONE" and self.candidate_hold_seconds >= CUPPED_ENTER_HOLD_SECONDS:
                self.pose = "CUPPED_HANDS"
        else:
            self.candidate_start_time = None
            self.candidate_hold_seconds = 0.0
            if self.pose == "CUPPED_HANDS":
                if self.failure_start_time is None:
                    self.failure_start_time = now
                self.failure_hold_seconds = max(0.0, now - self.failure_start_time)
            else:
                self.failure_start_time = None
                self.failure_hold_seconds = 0.0
            if self.pose == "CUPPED_HANDS" and self.failure_hold_seconds >= CUPPED_EXIT_HOLD_SECONDS:
                self.pose = "NONE"
        return self.pose

    @property
    def candidate_hold_ms(self) -> int:
        return int(self.candidate_hold_seconds * 1000.0)


def palm_center(landmarks) -> tuple[float, float]:
    points = [landmarks[index] for index in PALM_LANDMARK_INDICES]
    return (
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def distance(first, second) -> float:
    return hypot(first.x - second.x, first.y - second.y)


def official_gesture(result, index: int) -> tuple[str, float]:
    if not result.gestures or index >= len(result.gestures) or not result.gestures[index]:
        return "None", 0.0
    category = result.gestures[index][0]
    return category.category_name or "None", float(category.score or 0.0)


def handedness_label(result, index: int) -> str:
    if not result.handedness or index >= len(result.handedness):
        return "None"
    categories = result.handedness[index]
    if not categories:
        return "None"
    return categories[0].category_name or "None"


def make_hand_infos(result) -> list[HandInfo]:
    infos: list[HandInfo] = []
    for index, landmarks in enumerate(result.hand_landmarks):
        palm_x, palm_y = palm_center(landmarks)
        gesture, confidence = official_gesture(result, index)
        infos.append(
            HandInfo(
                landmarks=landmarks,
                palm_x=palm_x,
                palm_y=palm_y,
                official_gesture=gesture,
                confidence=confidence,
                handedness=handedness_label(result, index),
            )
        )
    # The labels are screen positions, not biological left/right hands. This
    # avoids assumptions about preview mirroring or result-array ordering.
    return sorted(infos, key=lambda hand: hand.palm_x)


def is_closed_fist(hand: HandInfo) -> bool:
    return hand.official_gesture == "Closed_Fist"


def open_finger_count(hand: HandInfo) -> int:
    landmarks = hand.landmarks
    wrist = landmarks[0]
    count = 0
    # Four fingers are enough for the first version; thumb geometry varies
    # more with a cupped pose and is therefore not required here.
    for tip_index, mcp_index in ((8, 5), (12, 9), (16, 13), (20, 17)):
        tip_distance = distance(landmarks[tip_index], wrist)
        mcp_distance = distance(landmarks[mcp_index], wrist)
        if tip_distance >= mcp_distance * OPEN_FINGER_DISTANCE_RATIO:
            count += 1
    return count


def evaluate_pose(infos: list[HandInfo]) -> PoseChecks:
    if len(infos) != 2:
        return PoseChecks(False, False, False, False, False, False, False, False, 0.0, 0.0, 0.0)

    left, right = infos
    palm_distance = hypot(right.palm_x - left.palm_x, right.palm_y - left.palm_y)
    y_difference = abs(left.palm_y - right.palm_y)
    left_open = open_finger_count(left) >= MIN_OPEN_FINGERS and not is_closed_fist(left)
    right_open = open_finger_count(right) >= MIN_OPEN_FINGERS and not is_closed_fist(right)
    wrist_gap_x = abs(left.landmarks[0].x - right.landmarks[0].x)
    index_gap_x = abs(left.landmarks[8].x - right.landmarks[8].x)
    index_wrist_ratio = index_gap_x / wrist_gap_x if wrist_gap_x >= MIN_WRIST_GAP_X else 0.0
    return PoseChecks(
        two_hands=True,
        distance_ok=MIN_PALM_DISTANCE <= palm_distance <= MAX_PALM_DISTANCE,
        height_ok=y_difference <= MAX_PALM_Y_DIFF,
        separation_ok=(right.palm_x - left.palm_x) >= MIN_HORIZONTAL_GAP,
        left_open=left_open,
        right_open=right_open,
        spread_shape_ok=(
            wrist_gap_x >= MIN_WRIST_GAP_X
            and index_wrist_ratio >= MIN_INDEX_WRIST_RATIO
        ),
        fists_rejected=not is_closed_fist(left) and not is_closed_fist(right),
        wrist_gap_x=wrist_gap_x,
        index_gap_x=index_gap_x,
        index_wrist_ratio=index_wrist_ratio,
    )


def draw_text(frame, lines: list[str], cv2) -> None:
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (12, 26 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


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
        num_hands=2,
    )
    stabilizer = PoseStabilizer()
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

                infos = make_hand_infos(result)
                for hand in infos:
                    draw_hand(frame, hand.landmarks, cv2)

                checks = evaluate_pose(infos)
                pose = stabilizer.update(now, checks.candidate)
                hands_count = len(infos)
                lines = [
                    f"FPS: {fps:5.1f}",
                    f"Hands: {hands_count} / 2",
                    f"Screen-left palm: {infos[0].palm_x:.3f}, {infos[0].palm_y:.3f}" if hands_count >= 1 else "Screen-left palm: --",
                    f"Screen-right palm: {infos[1].palm_x:.3f}, {infos[1].palm_y:.3f}" if hands_count >= 2 else "Screen-right palm: --",
                    f"Palm distance: {hypot(infos[1].palm_x - infos[0].palm_x, infos[1].palm_y - infos[0].palm_y):.3f}" if hands_count == 2 else "Palm distance: --",
                    f"Palm y difference: {abs(infos[0].palm_y - infos[1].palm_y):.3f}" if hands_count == 2 else "Palm y difference: --",
                    f"Wrist gap X: {checks.wrist_gap_x:.3f}",
                    f"Index gap X: {checks.index_gap_x:.3f}",
                    f"Index/Wrist ratio: {checks.index_wrist_ratio:.2f}",
                ]
                if hands_count >= 1:
                    lines.append(
                        f"Left-screen official: {infos[0].official_gesture} ({infos[0].handedness})"
                    )
                if hands_count >= 2:
                    lines.append(f"Right-screen official: {infos[1].official_gesture} ({infos[1].handedness})")
                lines.extend(
                    [
                        f"Two hands: {'YES' if checks.two_hands else 'NO'}",
                        f"Distance OK: {'YES' if checks.distance_ok else 'NO'}",
                        f"Height OK: {'YES' if checks.height_ok else 'NO'}",
                        f"Left hand open enough: {'YES' if checks.left_open else 'NO'}",
                        f"Right hand open enough: {'YES' if checks.right_open else 'NO'}",
                        f"Spread shape OK: {'YES' if checks.spread_shape_ok else 'NO'}",
                        f"Pose candidate: {'YES' if checks.candidate else 'NO'}",
                        f"Candidate hold: {stabilizer.candidate_hold_ms} ms / {int(CUPPED_ENTER_HOLD_SECONDS * 1000)} ms",
                        f"Pose: {pose}",
                        "Press Q to quit",
                    ]
                )
                draw_text(frame, lines, cv2)

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
