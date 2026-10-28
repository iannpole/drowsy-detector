from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "drowsy_model.pkl"
TASK_PATH = ROOT / "models" / "face_landmarker.task"
STATIC_DIR = Path(__file__).resolve().parent / "static"
PERCLOS_THRESHOLD = 0.2
YAW_THRESHOLD = 20.0


class FrameMetrics(BaseModel):
    ear: float = Field(ge=0, le=1)
    perclos: float = Field(ge=0, le=1)
    yaw_mean: float = Field(ge=-180, le=180)
    pitch_mean: float = Field(ge=-180, le=180)
    baseline: float = Field(ge=0.1, le=0.5)
    mar_mean: float = Field(ge=0, le=5)


def load_bundle():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
app = FastAPI()


def quick_detect(m: FrameMetrics):
    """Fast detection on individual frame metrics."""
    if m.perclos > PERCLOS_THRESHOLD:
        return "drowsy"
    if abs(m.yaw_mean) > YAW_THRESHOLD:
        return "distracted"
    if m.ear < m.baseline * 0.7:
        return "drowsy"
    return "alert"


def predict(metrics: FrameMetrics):
    status = quick_detect(metrics)
    confidence = None
    model_name = "quick"
    
    if bundle and bundle.get("estimator"):
        names = bundle["features"]
        row = pd.DataFrame([[
            metrics.ear, 0.01, metrics.ear / metrics.baseline,
            metrics.mar_mean, metrics.perclos, 0, 0,
            metrics.pitch_mean, 0, metrics.yaw_mean, 0
        ]], columns=names)
        estimator = bundle["estimator"]
        status = str(estimator.predict(row)[0])
        if hasattr(estimator, "predict_proba"):
            confidence = float(estimator.predict_proba(row)[0].max())
        model_name = bundle["model_name"]
    
    return status, confidence, model_name


@app.websocket("/ws")
async def monitor(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_json()
                metrics = FrameMetrics.model_validate(data)
            except ValueError:
                await websocket.send_json({"error": "invalid data"})
                continue
            status, confidence, model = predict(metrics)
            await websocket.send_json({"status": status, "confidence": confidence, "model": model})
    except WebSocketDisconnect:
        pass


@app.get("/model/face_landmarker.task")
def face_landmarker_task():
    if not TASK_PATH.is_file():
        return FileResponse(TASK_PATH, status_code=404)
    return FileResponse(TASK_PATH, media_type="application/octet-stream")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
