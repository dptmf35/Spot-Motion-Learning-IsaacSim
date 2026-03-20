# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous gait data collection system for Boston Dynamics Spot robot in Isaac Sim 5.0. Runs pre-trained RL policies to collect joint states, contact forces, trajectories, and ZMP data for supervised learning of inverse kinematics and motion.

## Commands

### Running the Main Application

Requires Isaac Sim 5.0's bundled Python interpreter:

```bash
~/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh \
  applications/spot_sensors_nav_data_collection.py \
  --num-episodes 100 \
  --episode-duration 60.0 \
  --dashboard-port 8001
```

Key CLI flags: `--num-episodes`, `--episode-duration`, `--spawn-x/y/yaw`, `--arena-x/y-min/max`, `--pos-thresh`, `--yaw-thresh`, `--dashboard-port`, `--no-auto-start`

### Frontend Development

```bash
cd dashboard/frontend
npm install
npm run dev      # Dev server at localhost:5173
npm run build    # Production build → dist/
```

### Testing Contact Sensor Bridge

```bash
~/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh tools/test_contact_bridge.py
# Output → /tmp/diag_output.txt
```

### Dashboard Access

Navigate to `http://localhost:8001` while the simulator is running.

## Architecture

### Dual-Policy Hierarchical Control

Two TorchScript policies in `policies/` run at different frequencies:
- **`SpotNavigationPolicy`** (5 Hz): Takes goal position → outputs velocity command `[vx, vy, wz]`
- **`SpotFlatTerrainPolicy`** (50 Hz): Takes 48-dim observation → outputs 12 joint position commands

Physics simulation runs at 500 Hz (dt=0.002). The gait policy runs every physics step; nav policy runs every 100 steps.

### Threading Model

Three concurrent execution contexts:
1. **Isaac Sim main thread** — physics stepping, policy inference, contact sensor collection; pushes state to `StateManager` queue
2. **FastAPI async server** — REST + WebSocket endpoints; pops from `StateManager`, broadcasts to clients
3. **Browser** — React dashboard receiving WebSocket state stream

`dashboard/backend/state_bridge.py` is the non-blocking bridge between (1) and (2) — drops oldest states on overflow, maintains latest-state cache.

### Episode Lifecycle

Episodes do **not** call `world.reset()` between them (expensive). Instead, the robot is teleported via `set_world_pose()` + `set_joint_positions()`. Only the very first episode or after a pause/resume triggers a full reset.

Timesteps are buffered in memory during an episode, then atomically flushed to HDF5 on completion. This prevents partial datasets.

### Contact Sensing

`dashboard/backend/contact_sensor_bridge.py` uses Isaac Sim 5.0's native `isaacsim.sensors.physics.ContactSensor` via `PhysxContactReportAPI`. Sensor USD prims must be created **before** `world.reset()` for proper initialization. Collects per-foot `[Fx, Fy, Fz]` forces and computes ZMP.

### HDF5 Output Schema

One HDF5 file per session at `data/gait_recordings/gait_session_YYYYMMDD_HHMMSS.h5`, with one dataset per episode.

Per-timestep fields: `timestamps`, `obs` (48,), `actions` (12,), `commands` (3,), `body_pose` (7,), `body_lin_vel` (3,), `body_ang_vel` (3,), `com_position` (3,), `zmp_position` (2,), `foot_positions` (4,3), `foot_contact_forces` (4,3), `foot_air_time` (4,), `head_position` (3,), `joint_pos` (12,), `joint_vel` (12,)

### Key Files

| File | Purpose |
|------|---------|
| `applications/spot_sensors_nav_data_collection.py` | Main entry point; `SpotGaitDataCollector`, `RandomWaypointGenerator`, `AutoCollectionManager` |
| `applications/spot_policy.py` | Policy loading and inference wrappers |
| `dashboard/backend/gait_main.py` | FastAPI app; `/api/status`, `/api/collection/*` REST routes + WebSocket |
| `dashboard/backend/gait_recorder.py` | HDF5 episode recording |
| `dashboard/backend/contact_sensor_bridge.py` | Physics contact sensor integration and ZMP computation |
| `dashboard/backend/state_bridge.py` | Thread-safe state queue (sim → API) |
| `dashboard/frontend/src/tabs/GaitDataCollection.tsx` | Main dashboard UI (recharts, real-time state) |
| `assets/spot_sensors.usd` | Spot robot USD with pre-attached contact sensor primitives |
| `policies/spot_flat/params/env.yaml` | Gait policy observation/action dimensions and normalization |
| `policies/spot_nav/params/env.yaml` | Navigation policy config |

## Important Constraints

- **Always use Isaac Sim's Python interpreter** (`python.sh`), not system Python. Standard `python3` lacks Isaac Sim bindings.
- Contact sensor prims must be added to the USD stage before `world.reset()` is called — order matters.
- The frontend (`dashboard/frontend/dist/`) is served statically by FastAPI when built. During development, Vite proxy handles API calls to the backend.
- `data/gait_recordings/*.h5` files are git-ignored (large binary output).
