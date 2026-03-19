"""GaitRecorder: extended HDF5 recorder with full gait data schema."""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


class GaitRecorder:
    """
    Episode-based HDF5 recorder for gait reference data collection.

    Extended schema vs baseline HDF5Recorder:
    - Added: body_lin_vel, body_ang_vel, com_position, zmp_position,
             foot_positions, foot_contact_forces, foot_air_time,
             head_position, joint_pos, joint_vel
    - Renamed: poses -> body_pose, action -> actions
    """

    # Field definitions: key -> (shape_suffix, dtype)
    # shape_suffix is the shape per timestep (N is prepended automatically)
    FIELDS = {
        "timestamps":           ((),        np.float64),
        "obs":                  ((48,),     np.float32),
        "actions":              ((12,),     np.float32),
        "commands":             ((3,),      np.float32),
        "body_pose":            ((7,),      np.float32),
        "body_lin_vel":         ((3,),      np.float32),
        "body_ang_vel":         ((3,),      np.float32),
        "com_position":         ((3,),      np.float32),
        "zmp_position":         ((2,),      np.float32),
        "foot_positions":       ((4, 3),    np.float32),
        "foot_contact_forces":  ((4, 3),    np.float32),
        "foot_air_time":        ((4,),      np.float32),
        "head_position":        ((3,),      np.float32),
        "joint_pos":            ((12,),     np.float32),
        "joint_vel":            ((12,),     np.float32),
    }

    def __init__(self, save_dir: str = "data/gait_recordings"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._file: Optional[h5py.File] = None
        self._file_path: Optional[Path] = None
        self._episode_count: int = 0
        self._recording: bool = False
        self._episode_start_time: Optional[float] = None
        self._episodes_meta: list = []
        self._buffer: dict = {}
        self._episode_waypoints: list = []
        self._arena_bounds: dict = {}

    def start_session(self) -> str:
        """Create a new HDF5 session file."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file_path = self.save_dir / f"gait_session_{ts}.h5"
        self._file = h5py.File(self._file_path, "w")
        self._file.attrs["created_at"] = ts
        self._file.attrs["robot_type"] = "SpotFlat"
        self._file.attrs["obs_dim"] = 48
        self._file.attrs["action_dim"] = 12
        self._episode_count = 0
        self._episodes_meta = []
        return str(self._file_path)

    def start_episode(self, arena_bounds: dict = None) -> int:
        """Begin buffering a new episode. Returns episode index."""
        if self._file is None:
            self.start_session()
        self._recording = True
        self._episode_start_time = time.time()
        self._episode_waypoints = []
        self._arena_bounds = arena_bounds or {}
        self._buffer = {key: [] for key in self.FIELDS}
        return self._episode_count

    def add_step(self, state: dict) -> None:
        """Buffer one timestep. state keys must match FIELDS."""
        if not self._recording:
            return
        for key in self.FIELDS:
            val = state.get(key)
            if val is not None:
                self._buffer[key].append(val)

    def record_waypoint(self, waypoint: tuple) -> None:
        """Record a visited waypoint (x, y, yaw) for episode metadata."""
        self._episode_waypoints.append(list(waypoint))

    def stop_episode(self) -> dict:
        """Flush buffer to HDF5 and return episode metadata."""
        if not self._recording or self._file is None:
            return {}

        self._recording = False
        end_time = time.time()
        duration = end_time - self._episode_start_time

        n_steps = len(self._buffer.get("timestamps", []))
        ep_name = f"episode_{self._episode_count:04d}"

        if n_steps > 0:
            grp = self._file.create_group(ep_name)
            for key, (shape_suffix, dtype) in self.FIELDS.items():
                buf = self._buffer.get(key, [])
                if not buf:
                    continue
                arr = np.array(buf, dtype=dtype)
                # Expected shape: (N, *shape_suffix) or (N,) for scalar
                grp.create_dataset(key, data=arr)

            grp.attrs["start_time"] = self._episode_start_time
            grp.attrs["end_time"] = end_time
            grp.attrs["duration_sec"] = duration
            grp.attrs["n_steps"] = n_steps
            grp.attrs["episode_num"] = self._episode_count
            grp.attrs["waypoints"] = json.dumps(self._episode_waypoints)
            grp.attrs["arena_bounds"] = json.dumps(self._arena_bounds)
            self._file.flush()

        meta = {
            "id": self._episode_count,
            "name": ep_name,
            "n_steps": n_steps,
            "duration_sec": round(duration, 2),
            "start_time": datetime.fromtimestamp(self._episode_start_time).strftime("%H:%M:%S"),
            "waypoints_visited": len(self._episode_waypoints),
        }
        self._episodes_meta.append(meta)
        self._episode_count += 1
        self._buffer = {}
        return meta

    def write_buffered_episode(self, buffer: dict, start_time: float,
                               arena_bounds: dict = None, waypoints: list = None) -> dict:
        """Write a pre-buffered episode atomically (for multi-env use)."""
        if self._file is None:
            self.start_session()

        end_time = time.time()
        duration = end_time - start_time
        n_steps = len(buffer.get("timestamps", []))
        ep_name = f"episode_{self._episode_count:04d}"

        if n_steps > 0:
            grp = self._file.create_group(ep_name)
            for key, (shape_suffix, dtype) in self.FIELDS.items():
                buf = buffer.get(key, [])
                if not buf:
                    continue
                grp.create_dataset(key, data=np.array(buf, dtype=dtype))
            grp.attrs["start_time"] = start_time
            grp.attrs["end_time"] = end_time
            grp.attrs["duration_sec"] = duration
            grp.attrs["n_steps"] = n_steps
            grp.attrs["episode_num"] = self._episode_count
            grp.attrs["waypoints"] = json.dumps(waypoints or [])
            grp.attrs["arena_bounds"] = json.dumps(arena_bounds or {})
            self._file.flush()

        meta = {
            "id": self._episode_count,
            "name": ep_name,
            "n_steps": n_steps,
            "duration_sec": round(duration, 2),
            "start_time": datetime.fromtimestamp(start_time).strftime("%H:%M:%S"),
            "waypoints_visited": len(waypoints or []),
        }
        self._episodes_meta.append(meta)
        self._episode_count += 1
        return meta

    def close(self) -> None:
        if self._recording:
            self.stop_episode()
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def get_episodes(self) -> list:
        return list(self._episodes_meta)

    def get_file_size(self) -> int:
        if self._file_path and self._file_path.exists():
            return self._file_path.stat().st_size
        return 0

    def get_recording_duration(self) -> float:
        if self._recording and self._episode_start_time:
            return time.time() - self._episode_start_time
        return 0.0
