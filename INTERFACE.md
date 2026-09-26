# Camera Worker JSONL Interface

## 1. Scope and responsibility

`camera_worker.py` is a standalone visual-sensor process. It currently:

- opens the default camera (`cv2.VideoCapture(0)`);
- runs one MediaPipe Gesture Recognizer per frame;
- reads the official gesture result and the same result's hand landmarks;
- calculates palm position, motion, WAVE, STILL, and silence state;
- writes machine-readable JSON Lines to stdout.

It does not implement game story/scene logic, Qt UI, menu logic, game control,
or serial communication. Those belong to the consuming application.

The worker uses only `models\gesture_recognizer.task`; it does not create a
second Hand Landmarker.

## 2. Process model

The intended consumer is a Qt/C++ process using `QProcess`:

```text
Qt application -> QProcess -> .venv\Scripts\python.exe camera_worker.py
                                      |
                                      +-> stdout JSONL
                                      +-> stderr diagnostics
```

The default worker is headless. `--preview` is an optional local debugging
mode; it adds an OpenCV window but does not change stdout into anything other
than JSONL.

## 3. JSONL rules

- Every emitted protocol message is one complete JSON object on one line.
- Every protocol message contains `v` and `ts_ms`.
- Current protocol version is `v: 1`.
- `ts_ms` is an integer monotonic timestamp in milliseconds relative to worker
  startup, not a wall-clock date/time.
- `stdout` is reserved for JSONL. Do not merge stderr into stdout in the Qt
  process.
- Native MediaPipe warnings and worker diagnostics go to `stderr` and must not
  be parsed as protocol messages.
- The reader must buffer stdout and split on newline; one `readyRead` callback
  is not guaranteed to contain exactly one JSON object.

## 4. Message types

| `type` | Kind | Current emission behavior |
|---|---|---|
| `CAMERA_STATE` | State | Startup success/failure and camera read failure. |
| `GESTURE` | State change | Initial state and changes among `PALM`, `GOOD`, and `NONE`. |
| `PALM` | Continuous state | About 10 Hz while a hand is present. |
| `WAVE` | Discrete event | One message when WAVE becomes active; no per-frame banner. |
| `SILENCE_STATE` | Continuous state | About 10 Hz and immediately when state changes. |
| `SILENCE_REACHED` | Discrete event | Once when valid continuous stillness reaches the current threshold. |

`GESTURE` and `PALM` are different messages: `GESTURE` reports a discrete
game-facing label, while `PALM` reports the normalized palm position.

## 5. Common fields

```json
{"v":1,"type":"WAVE","ts_ms":6300}
```

- `v`: protocol version, currently `1`.
- `type`: one of the message types listed above.
- `ts_ms`: worker-relative monotonic milliseconds.

## 6. Message schemas and meanings

### CAMERA_STATE

```json
{"v":1,"type":"CAMERA_STATE","available":true,"ts_ms":1557}
```

- `available: true`: Gesture Recognizer loaded and camera opened.
- `available: false`: camera/model startup failed or a camera frame could not
  be read.
- On a normal startup, the first protocol message is `available: true`.
- If the Python import or model path check fails before worker initialization,
  the current code reports the error on stderr and exits before emitting a
  protocol message.

### GESTURE

```json
{"v":1,"type":"GESTURE","gesture":"PALM","confidence":0.94,"ts_ms":850}
```

- `gesture` is the worker label, not the raw MediaPipe category:
  - `Open_Palm` -> `PALM`
  - `Thumb_Up` -> `GOOD`
  - any other category or no hand -> `NONE`
- `confidence` is the official MediaPipe category score. With no hand it is
  `0.0`.
- The worker sends this only when the worker label changes, rather than once
  per camera frame.

### PALM

```json
{"v":1,"type":"PALM","x":0.43,"y":0.58,"ts_ms":860}
```

- `x` and `y` are normalized image coordinates.
- They are calculated as the arithmetic mean of landmarks `0, 5, 9, 13, 17`
  from the same Gesture Recognizer result.
- The current code does not explicitly clamp the values to `[0, 1]`; the
  model normally returns normalized coordinates, but an edge-of-frame value
  may be slightly outside that range. This is a pending integration detail,
  not a promise of clamped output.
- `PALM` is emitted at approximately 10 Hz while a hand is detected. No fake
  `x: 0, y: 0` message is emitted for `NO_HAND`.

### WAVE

```json
{"v":1,"type":"WAVE","ts_ms":1300}
```

This is a discrete event. It is emitted on the inactive-to-active WAVE
transition. Holding the WAVE banner active does not produce repeated messages;
the detector must later satisfy its own conditions again before another event.

### SILENCE_STATE

```json
{"v":1,"type":"SILENCE_STATE","motion":0.043,"still_ms":1280,"state":"STILL","ts_ms":9200}
```

- `motion`: current smoothed overall hand motion from the existing tracker.
- `still_ms`: valid continuous stillness time in milliseconds.
- `state` is exactly one of:
  - `MOVING`
  - `STILL`
  - `NO_HAND`
- `NO_HAND` is not a successful stillness result. It clears the timer and
  motion history and does not automatically unlock a previously emitted
  silence event.
- `MOVING` clears `still_ms`; valid `STILL` intervals add their actual frame
  time; invalid tracking intervals do not count.

### SILENCE_REACHED

```json
{"v":1,"type":"SILENCE_REACHED","ts_ms":3400}
```

This is a one-shot event for a stillness period. Continuous valid STILL time
must reach the current internal threshold (currently about 2000 ms). Continued
stillness does not repeat it. Clear MOVING or WAVE unlocks the next stillness
period; `NO_HAND` alone does not.

## 7. Qt/QProcess integration guidance

The consuming application should:

1. Start the project interpreter with the absolute project working directory:
   `.venv\Scripts\python.exe camera_worker.py`.
2. Keep stdout and stderr as separate channels.
3. Append stdout chunks to a buffer and split complete newline-delimited JSONL
   records.
4. Parse each complete line with `QJsonDocument::fromJson()`.
5. Dispatch by `type`:
   - `CAMERA_STATE`: update visual-input availability;
   - `GESTURE`: update the current discrete gesture;
   - `PALM`: update palm position;
   - `WAVE`: handle one wave event;
   - `SILENCE_STATE`: update continuous movement state;
   - `SILENCE_REACHED`: handle one completed stillness event.
6. Treat `CAMERA_STATE available:false` or worker exit as invalid visual input;
   do not keep using the last gesture, palm, or STILL value as current data.

This document describes the current Python worker only; it is not a complete
Qt implementation.
