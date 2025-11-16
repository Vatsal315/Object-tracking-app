## Object Tracking and Detection (OpenCV + YOLOv4)

This project shows a compact pipeline that detects vehicles with YOLOv4 (via OpenCV DNN) and tracks them across frames using a lightweight centroid‑ID tracker. It’s designed to be easy to run on CPU and will automatically use CUDA when available.

### Highlights
- YOLOv4 inference through OpenCV’s `cv2.dnn` API (no Darknet build required)
- Simple and readable centroid tracker that assigns persistent IDs
- Automatic CUDA→CPU fallback (runs on macOS out of the box)
- Works on a sample video (`los_angeles.mp4`), and supports any other video path

## Requirements
- Python 3.10+
- Packages: `opencv-python`, `numpy`

## Quickstart

```bash
# from project root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install opencv-python numpy

# download YOLOv4 weights (~245 MB)
curl -L -o dnn_model/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights

# run the demo
python object_tracking.py
```

- A window titled “Frame” will open. Press Esc to quit.
- By default the model input size is 608. For faster CPU runs, change `image_size` to `416` in `object_detection.py`.

## Project structure
- `object_detection.py`: YOLOv4 loader and a small wrapper with NMS/conf thresholds
- `object_tracking.py`: video loop, detection calls, centroid-based ID tracking and drawing
- `dnn_model/`
  - `yolov4.cfg`: model config
  - `classes.txt`: 80 COCO class names
  - `yolov4.weights`: download at first run (not tracked in git)

## How it works (short version)
1. Each frame is passed through YOLOv4 to obtain bounding boxes.
2. For every box we compute its center point and maintain a mapping `object_id → center`.
3. Existing objects are matched to current centers by Euclidean distance. If a previous center is close to a new one, we update the same ID; otherwise, stale IDs are removed and new IDs are created.

This keeps IDs stable while objects are visible, and drops them when they leave or are not detected anymore.

<img width="960" height="568" alt="Screenshot 2025-11-15 at 6 09 08 PM" src="https://github.com/user-attachments/assets/b268c116-8b2e-4fec-a808-82ee77ead269" />


## Configuration knobs
- In `object_detection.py`:
  - `confThreshold` (default 0.5)
  - `nmsThreshold` (default 0.4)
  - `image_size` (default 608; set 416 for speed on CPU)
  - Backend/target is chosen automatically. If CUDA is unavailable, the code prints: “CUDA not available; using CPU backend for OpenCV DNN”.

## Troubleshooting
- OpenCV DNN CUDA assertion on macOS:
  - Fixed in this repo by proactively selecting CPU when `cv2.cuda.getCudaEnabledDeviceCount() == 0`.
- No window appears:
  - Ensure you run in a desktop session (not headless), and execute `python object_tracking.py` from the project root.

## Notes
- This repository intentionally excludes large/binary assets (video files, `.weights`, virtual environments). See `.gitignore` for details.
- YOLOv4 weights and config originate from the official Darknet releases by the YOLO authors.

## License and purpose
This codebase is provided for learning and experimentation. Use responsibly and verify performance and accuracy for your use case.
