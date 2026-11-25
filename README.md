# Drowsy Detector

A local drowsiness-monitoring prototype using MediaPipe Face Landmarker. The browser processes webcam frames locally and sends only numeric features to a FastAPI WebSocket. Video and images are never sent to or saved by the server.

## Setup

From the project directory on Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The required MediaPipe model must exist at:

```text
models/face_landmarker.task
```

## Run the web monitor

From `C:\VIscom project\drowsy-detector`:

```powershell
.venv\Scripts\python.exe -m uvicorn web.app:app --reload
```

Open this exact local URL in a current browser:

```text
http://localhost:8000
```

Do not open `index.html` directly as a file. `getUserMedia` works only in a secure context: `localhost` during local development or HTTPS in deployment.

### Manual steps

1. Allow camera access when the browser asks. Microphone access is not requested.
2. Sit at the intended monitoring distance with stable frontal lighting.
3. Click **Calibrate (10 s)** and keep your eyes naturally open while looking forward.
4. After calibration completes, click **Start**.
5. Interact with the page once before expecting alarm sounds; browsers require a user gesture to enable Web Audio. The Calibrate button supplies that gesture.
6. Click **Stop** when finished, then close the tab or revoke camera permission if desired.

During first load, the page needs internet access to fetch the pinned `@mediapipe/tasks-vision` JavaScript and WebAssembly files from jsDelivr. The face model itself is served locally from `models/face_landmarker.task`.

The backend loads `models/drowsy_model.pkl` when it starts. Restart Uvicorn after retraining so the new model is loaded. If the model file is missing, the server falls back to PERCLOS and yaw threshold rules.

## Data collection

```powershell
.venv\Scripts\python.exe src\collect.py --subject person01
```

Controls: `0` alert, `1` drowsy, `2` distracted, Space pause/resume, and `q` quit. Only numeric CSV features are stored.

Dry run without a camera:

```powershell
.venv\Scripts\python.exe src\collect.py --subject test --out data\test.csv --dry-run
```

## External datasets

Two public image datasets are used. Download them manually (Kaggle login required) and extract them to:

- `data/external_datasets/nthu/images/` from https://www.kaggle.com/datasets/ahmadrizalmustofa/nthu-multiclass (7,500 images, subjects `001`, `002`, labels from filename `*_drowsy.jpg` / `*_notdrowsy.jpg`)
- `data/external_datasets/uta/images/` from https://www.kaggle.com/datasets/minhngt02/uta-rldd (11,787 images, `train|val|test/{active,fatigue}/`)

Extract features (EAR, MAR, head pose) with MediaPipe. The output is a CSV in the pipeline format:

```powershell
.venv\Scripts\python.exe src\extract_features.py nthu data\external_datasets\nthu\images --out data\nthu.csv
.venv\Scripts\python.exe src\extract_features.py uta data\external_datasets\uta\images --out data\uta.csv
```

Raw images are never committed; only the extracted numeric CSVs in `data/` are.

## Training

```powershell
.venv\Scripts\python.exe src\train.py
```

Training reads all `data/*.csv` files. The split is:

- **UTA-RLDD**: official `train` / `val` / `test` folders. `val` is merged into training; `test` is held out.
- **NTHU**: `GroupShuffleSplit` by subject (half of subjects held out).

Imbalance is handled with `class_weight='balanced'` for Random Forest and SVM. Results are written to `models/report.md`. Restart Uvicorn after retraining.
