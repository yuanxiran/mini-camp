# MediaPipe Face Actions Demo

This is a local Windows demo that reads the default camera, runs MediaPipe
Face Landmarker blendshape inference, and displays the values in an OpenCV
window. Frames, videos, and face features are not saved or uploaded.

## First run

From this project directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the official Face Landmarker model and save it as:

```text
models\face_landmarker.task
```

The official model URL is:

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Then run:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Later runs

```powershell
.\.venv\Scripts\python.exe main.py
```

Press `Q` to quit. The main program is `main.py`.

## Action thresholds

The first version treats `eyeBlinkLeft` and `eyeBlinkRight` as eye-closure
scores; it does not count blink events. Threshold constants are at the top of
`main.py`:

- Enter threshold: `0.55`
- Exit threshold: `0.45`
- Required consecutive frames: `3`

## Troubleshooting

If the program reports an error, send the complete PowerShell command and the
full error output. Do not send camera images or any personal data.

## Hand prototype

The independent hand prototype is `hand_demo.py`. It uses the official Hand
Landmarker model in `models\hand_landmarker.task` and does not change the Face
Demo in `main.py`.

Run it with:

```powershell
.\.venv\Scripts\python.exe hand_demo.py
```

The window shows one hand's skeleton, a palm center calculated as the average
of landmarks `0, 5, 9, 13, 17`, normalized `palm_x`/`palm_y` values, horizontal
motion, FPS, and `NO HAND`, `TRACKING`, or candidate `WAVE` status. WAVE uses
the recent `palm_x` trajectory (range, path length, and direction reversals),
not a single static frame.

`motion` is the five-sample average speed of the same five palm landmarks in
normalized image coordinates per second. Actual time between valid frames is
used. The first tracked frame, the first frame after `NO HAND`, and time gaps
outside `0.005` to `0.25` seconds only rebuild the tracking baseline.

The initial stillness defaults are:

- Enter `STILL` at `motion <= 0.08`
- Return to `MOVING` at `motion >= 0.15`
- Emit one `SILENCE_REACHED` after 2000 ms of valid continuous stillness

`still_ms` only accumulates valid tracked `STILL` frame intervals. `NO HAND`
clears tracking and the timer but does not unlock an already emitted silence
event. Clear movement or WAVE unlocks the next event. JSONL output and Qt
integration are intentionally not implemented yet.

## Gesture Recognizer prototype

The independent `gesture_demo.py` validates the official MediaPipe Gesture
Recognizer using `models\gesture_recognizer.task`. It displays the official
gesture category and score, draws the landmarks returned by that same
inference, and maps only these debug labels:

- `Open_Palm` -> `PALM`
- `Thumb_Up` -> `GOOD`
- Other or no result -> `NONE`

Run it with:

```powershell
.\.venv\Scripts\python.exe gesture_demo.py
```

This prototype does not modify `main.py` or `hand_demo.py`, does not implement
JSONL/Qt integration, and does not save or upload camera data.

## Unified hand prototype

`unified_hand_demo.py` validates Gesture Recognizer as the only hand inference
backend. The same inference result supplies official gestures and the hand
landmarks used by the existing palm, motion, WAVE, STILL, and SILENCE logic; it
does not create a separate Hand Landmarker.

Run it with:

```powershell
.\.venv\Scripts\python.exe unified_hand_demo.py
```

This remains an independent validation demo. It does not replace the existing
hand or gesture demos and does not implement Camera Worker, JSONL, or Qt.

## Camera Worker

`camera_worker.py` is the headless hand worker for a future Qt `QProcess`
integration. It runs only Gesture Recognizer, reads gestures and hand
landmarks from the same result, and writes one JSON object per stdout line.
Human-readable errors go to stderr; stdout must remain JSONL-only.

Run the worker:

```powershell
.\.venv\Scripts\python.exe camera_worker.py
```

Run the optional local preview (stdout is still JSONL-only):

```powershell
.\.venv\Scripts\python.exe camera_worker.py --preview
```

Message types are `CAMERA_STATE`, `GESTURE`, `PALM`, `WAVE`,
`SILENCE_STATE`, and `SILENCE_REACHED`. Every message contains `v: 1` and a
worker-relative `ts_ms`. Gesture and WAVE messages are emitted on state/event
changes; PALM and SILENCE_STATE are rate-limited to about 10 Hz. The worker
does not include Face, Qt, game control, or camera-data storage.

Documentation:

- [INTERFACE.md](INTERFACE.md): current JSONL fields and QProcess integration notes
- [USAGE.md](USAGE.md): first-checkout setup, commands, expected behavior, and troubleshooting
