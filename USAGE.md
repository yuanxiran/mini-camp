# Usage Guide

This guide is for a first checkout on Windows. The project contains several
visual validation demos and one formal headless worker. `camera_worker.py` is
the process entry point intended for Qt/C++ integration; the demos remain
useful for local diagnosis.

## 1. Project directory

Open PowerShell and enter the project directory:

```powershell
cd D:\my_code\AI_project\mini_camp\cv_face_test
```

All commands below are project-local. Do not use a global `pip`, `pip --user`,
or a system-wide Python installation for this project.

## 2. Prepare the Python environment

The current development environment was verified with Python 3.12.10. This is
an environment note, not a claim that other Python versions are supported.

If `.venv` does not exist in a fresh checkout, create it inside the project:

```powershell
py -3.12 -m venv .venv
```

Install the declared dependencies into that environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The current runtime dependencies are `mediapipe`, `opencv-python`, and
`numpy`. This documentation turn did not install anything.

## 3. Model files

The worker requires this model at the exact project-local path:

```text
models\gesture_recognizer.task
```

The standalone demos additionally use their own models:

```text
models\face_landmarker.task
models\hand_landmarker.task
```

Model files are ignored by the project `.gitignore`, so a fresh checkout may
need the approved official model-preparation step before running the demos.
This document does not perform downloads.

## 4. Run the formal worker

Headless mode is the default:

```powershell
.\.venv\Scripts\python.exe camera_worker.py
```

Expected normal startup begins with a JSONL record similar to:

```json
{"v":1,"type":"CAMERA_STATE","available":true,"ts_ms":20}
```

The terminal then receives JSONL records. That is expected worker output, not
debug noise. Human-readable errors and MediaPipe diagnostics belong to stderr.

Stop headless mode with `Ctrl+C`. The worker releases the camera and closes the
Gesture Recognizer in its cleanup path.

## 5. Run the preview mode

For local visual testing:

```powershell
.\.venv\Scripts\python.exe camera_worker.py --preview
```

Preview adds an OpenCV window showing landmarks, official gesture, confidence,
motion, state, stillness time, and FPS. Stdout remains JSONL-only. Press `Q`
or close the preview window with `X` to exit.

## 6. What to check

With a usable camera, check these behaviors in order:

1. Startup emits `CAMERA_STATE available:true`.
2. An open palm emits `GESTURE` with `gesture: "PALM"`.
3. A thumbs-up emits `GESTURE` with `gesture: "GOOD"`.
4. Moving the hand changes `PALM.x`/`PALM.y` and reports `MOVING`.
5. A left-right wave emits one `WAVE` event.
6. Holding the hand still changes `SILENCE_STATE` to `STILL` and increases
   `still_ms`.
7. Reaching the internal stillness duration emits one `SILENCE_REACHED`.
8. Removing the hand emits `GESTURE NONE` when the gesture changes and
   `SILENCE_STATE state: "NO_HAND"`; it does not count as STILL.
9. With two hands, a stable cupped pose emits one `TWO_HAND_POSE` message with
   `pose: "CUPPED_HANDS"`; continued holding does not repeat it.
10. Releasing the pose long enough emits one `TWO_HAND_POSE` message with
    `pose: "NONE"`.

The current worker has not been confirmed with a real camera in this execution
environment. Do not treat a successful model-load check as camera validation.

## 7. Common problems

### Camera cannot open

The worker emits `CAMERA_STATE available:false` when initialization or frame
reading fails and writes a human-readable explanation to stderr. Check whether
another application is using the camera and review Windows camera permission;
do not change system settings automatically.

During an already running worker, one failed frame read is treated as a short
transient gap. After three consecutive failed reads the worker emits
`CAMERA_STATE available:false`, stops emitting frame-derived messages, clears
old tracking state, and waits for a future successful frame. A successful frame
emits `CAMERA_STATE available:true` and resumes with a fresh baseline. This is
not a full camera reconnect manager; persistent camera failures should be
diagnosed at the device/permission level.

The worker also has limited recovery for an occasional MediaPipe Gesture
Recognizer graph runtime failure. It reports diagnostics only on stderr,
clears all visual continuity state, recreates the same `num_hands=2` video
recognizer, and emits `CAMERA_STATE available:false` followed by
`available:true` around the recovery. Restart attempts are bounded; repeated
failures cause a safe exit instead of an infinite restart loop. This recovery
does not change the JSONL protocol or claim that an underlying MediaPipe
version issue is solved.

### Model file not found

Confirm the exact path `models\gesture_recognizer.task` from the project root.
The worker exits with an error if the file is absent.

### MediaPipe warnings appear in the terminal

These are diagnostics on stderr. A Qt consumer must keep stderr separate from
stdout and parse only complete stdout JSONL lines.

### No JSON arrives immediately

The process may still be loading the model or opening the camera. Check stderr
and wait for `CAMERA_STATE`; do not assume that one stdout read equals one JSON
record.

### Two hands and primary-hand continuity

The formal worker runs Gesture Recognizer with `num_hands=2`. Existing
single-hand messages continue to follow one selected primary hand; they are not
duplicated into left/right streams. The worker follows that hand by nearest
palm position when result-array order changes. If the primary disappears or a
different hand cannot be matched safely, it rebuilds the motion/WAVE baseline
before calculating new values.

`TWO_HAND_POSE` is separate from `GESTURE`. It reports only `CUPPED_HANDS` or
`NONE`, using the validated two-hand geometry and real-time hold intervals. The
standalone `cupped_hands_demo.py` remains available when detailed threshold
diagnostics are needed.

## 8. Existing standalone demos

The existing demos remain independent:

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe hand_demo.py
.\.venv\Scripts\python.exe gesture_demo.py
.\.venv\Scripts\python.exe unified_hand_demo.py
```

`camera_worker.py` does not replace these debugging entry points. It is the
headless JSONL process for later integration.

## 9. Files a checkout should understand

| File | Purpose |
|---|---|
| `camera_worker.py` | Formal headless Gesture Recognizer + JSONL worker |
| `unified_hand_demo.py` | Unified visual debug demo |
| `gesture_demo.py` | Gesture Recognizer-only visual demo |
| `hand_demo.py` | Hand Landmarker/motion visual demo |
| `main.py` | Face demo |
| `cupped_hands_demo.py` | Detailed two-hand CUPPED_HANDS threshold demo |
| `models\` | Local `.task` model files |
| `requirements.txt` | Python dependencies |
| `INTERFACE.md` | JSONL contract for an integration consumer |
| `README.md` | Short project overview |
