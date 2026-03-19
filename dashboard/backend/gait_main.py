"""FastAPI server for gait data collection dashboard."""
import asyncio
import threading
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .state_bridge import StateManager
from .gait_recorder import GaitRecorder

app = FastAPI(title="Gait Data Collection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated by start_gait_dashboard_server()
_state_manager: Optional[StateManager] = None
_recorder: Optional[GaitRecorder] = None
_collection_manager = None   # AutoCollectionManager instance (set externally)


# ─────────────────────────── REST endpoints ────────────────────────────

@app.get("/api/status")
def get_status():
    col_status = {}
    if _collection_manager is not None:
        col_status = _collection_manager.get_status()
    return {
        "recording": _recorder.is_recording if _recorder else False,
        "episode_count": _recorder.episode_count if _recorder else 0,
        "recording_duration": _recorder.get_recording_duration() if _recorder else 0.0,
        "file_size_bytes": _recorder.get_file_size() if _recorder else 0,
        "connected_to_sim": _state_manager is not None and _state_manager.get_latest_state() is not None,
        "collection": col_status,
    }


@app.post("/api/collection/start")
def collection_start(body: dict = None):
    if _collection_manager is None:
        return JSONResponse({"error": "collection manager not ready"}, status_code=503)
    config = body or {}
    _collection_manager.start_collection(config)
    return {"status": "started", "config": _collection_manager.get_config()}


@app.post("/api/collection/stop")
def collection_stop():
    if _collection_manager is None:
        return JSONResponse({"error": "collection manager not ready"}, status_code=503)
    _collection_manager.stop_collection()
    return {"status": "stopped"}


@app.get("/api/collection/config")
def get_collection_config():
    if _collection_manager is None:
        return {}
    return _collection_manager.get_config()


@app.put("/api/collection/config")
def update_collection_config(body: dict):
    if _collection_manager is None:
        return JSONResponse({"error": "collection manager not ready"}, status_code=503)
    _collection_manager.update_config(body)
    return {"status": "updated", "config": _collection_manager.get_config()}


@app.get("/api/episodes")
def get_episodes():
    if _recorder is None:
        return []
    return _recorder.get_episodes()


# ─────────────────────────── WebSocket ────────────────────────────────

def _nan_to_null(obj):
    """Recursively replace float NaN/Inf with None for JSON safety."""
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    if isinstance(obj, list):
        return [_nan_to_null(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    return obj


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            state = _state_manager.get_latest_state() if _state_manager else None
            col_status = _collection_manager.get_status() if _collection_manager else {}
            if state:
                payload = _nan_to_null({
                    "timestamp": state.get("timestamp", 0),
                    "body_pose": state.get("body_pose", []),
                    "commands": state.get("commands", []),
                    "com_position": state.get("com_position", []),
                    "zmp_position": state.get("zmp_position", []),
                    "foot_positions": state.get("foot_positions", []),
                    "foot_contact": state.get("foot_contact", []),
                    "head_position": state.get("head_position", []),
                    "joint_pos": state.get("joint_pos", []),
                    "joint_vel": state.get("joint_vel", []),
                    "recording": _recorder.is_recording if _recorder else False,
                    "episode_count": _recorder.episode_count if _recorder else 0,
                    "recording_duration": _recorder.get_recording_duration() if _recorder else 0.0,
                    "collection": col_status,
                })
                await websocket.send_json(payload)
            await asyncio.sleep(0.1)  # 10 Hz
    except (WebSocketDisconnect, Exception):
        pass


# ─────────────────────────── Background recorder task ─────────────────

def _recorder_drain_loop(state_manager: StateManager, recorder: GaitRecorder):
    """Drain state_manager queue to prevent unbounded growth.
    Recording is now handled per-env in the physics callback."""
    while True:
        state_manager.drain_queue()   # drain only — recording done in physics callback
        time.sleep(0.1)


# ─────────────────────────── Server launcher ───────────────────────────

def set_collection_manager(manager) -> None:
    """Called from main app to inject AutoCollectionManager reference."""
    global _collection_manager
    _collection_manager = manager


def start_gait_dashboard_server(
    state_manager: StateManager,
    recorder: GaitRecorder,
    collection_manager=None,
    port: int = 8001,
) -> None:
    """Launch gait dashboard FastAPI server in daemon thread."""
    global _state_manager, _recorder, _collection_manager
    _state_manager = state_manager
    _recorder = recorder
    if collection_manager is not None:
        _collection_manager = collection_manager

    path = recorder.start_session()
    print(f"[GaitDashboard] HDF5 session: {path}")

    drain_thread = threading.Thread(
        target=_recorder_drain_loop,
        args=(state_manager, recorder),
        daemon=True,
    )
    drain_thread.start()

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        loop="none",          # avoid asyncio event-loop conflict with Isaac Sim
    )
    server = uvicorn.Server(config)

    def _run(srv):
        try:
            srv.run()
        except Exception as e:
            import traceback
            print(f"[GaitDashboard] FATAL: uvicorn crashed on port {port}: {e}", flush=True)
            traceback.print_exc()

    server_thread = threading.Thread(target=_run, args=(server,), daemon=True)
    server_thread.start()

    # Wait up to 5 s for uvicorn to actually bind before announcing ready
    import socket
    for _ in range(50):
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.1)
            s.close()
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)
    else:
        print(f"[GaitDashboard] WARNING: port {port} not responding after 5s — check for errors above")

    print(f"[GaitDashboard] API server running at http://localhost:{port}")
    print(f"[GaitDashboard] Open http://localhost:5173 in browser → 🦿 Gait Collection tab")
