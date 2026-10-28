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

Download dan integrate public datasets untuk training yang lebih robust:

```powershell
.venv\Scripts\python.exe src\download_datasets.py download
```

Dataset yang tersedia:
- **UTA-RLDD Processed Features** (~1.7M rows): 60 subjects, 3 kelas (alert/drowsy/distracted), dari University of Texas at Arlington
  - GitHub: https://github.com/Blueberrycheesecake069/uta-rldd-processed-features
  - Format: CSV dengan EAR, MAR, head pose

Script akan otomatis:
1. Download dataset eksternal ke `data/external_datasets/`
2. Konversi ke format pipeline (timestamp, subject, features, label)
3. Merge dengan data collection lokal Anda

Cek progress data:

```powershell
.venv\Scripts\python.exe src\validate_data.py
```

## Training

```powershell
.venv\Scripts\python.exe src\train.py
```

Training reads all `data/*.csv` dan `data/external_*.csv` files, evaluates dengan subject-grouped folds. Semakin banyak data & subject diversity = semakin akurat model.
