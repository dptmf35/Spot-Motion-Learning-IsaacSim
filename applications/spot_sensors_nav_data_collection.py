from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import argparse
import carb
import math
import numpy as np
import os
import random
import threading
import time
from pathlib import Path

import omni.appwindow
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.storage.native import get_assets_root_path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spot_policy import SpotFlatTerrainPolicy, SpotNavigationPolicy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))
from backend.state_bridge import StateManager
from backend.contact_sensor_bridge import ContactSensorBridge, compute_zmp
from backend.gait_recorder import GaitRecorder
from backend.gait_main import start_gait_dashboard_server

BASE_DIR = Path(__file__).resolve().parent.parent


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class RandomWaypointGenerator:
    """Generates random (x, y, yaw) goals within arena bounds."""

    def __init__(self, arena_bounds: dict, min_distance: float = 2.0,
                 max_distance: float = 8.0, max_retries: int = 50):
        self.bounds = arena_bounds
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_retries = max_retries

    def sample(self, current_pos: tuple) -> tuple:
        cx, cy = current_pos[0], current_pos[1]
        for _ in range(self.max_retries):
            x = random.uniform(self.bounds["x_min"], self.bounds["x_max"])
            y = random.uniform(self.bounds["y_min"], self.bounds["y_max"])
            dist = math.hypot(x - cx, y - cy)
            if self.min_distance <= dist <= self.max_distance:
                yaw = random.uniform(-math.pi, math.pi)
                return (x, y, yaw)
        x = random.uniform(self.bounds["x_min"], self.bounds["x_max"])
        y = random.uniform(self.bounds["y_min"], self.bounds["y_max"])
        yaw = random.uniform(-math.pi, math.pi)
        return (x, y, yaw)


class AutoCollectionManager:
    """Manages autonomous episode collection lifecycle."""

    def __init__(self, num_episodes: int, episode_duration_sec: float,
                 arena_bounds: dict, waypoint_generator: RandomWaypointGenerator,
                 recorder: "GaitRecorder"):
        self._num_episodes = num_episodes
        self._episode_duration = episode_duration_sec
        self._arena_bounds = arena_bounds
        self._wp_gen = waypoint_generator
        self._recorder = recorder
        self._lock = threading.Lock()

        self._collecting = False
        self._current_episode = 0
        self._episode_start_time = None
        self._goals_reached = 0
        self._current_wp = None
        self._world_reset_requested = False
        self._first_collection = True

    def start_collection(self, config: dict = None) -> None:
        with self._lock:
            if config:
                self._update_config_locked(config)
            self._collecting = True
            self._current_episode = 0
            if self._first_collection:
                self._first_collection = False
            else:
                self._world_reset_requested = True
            print(f"[AutoCollection] Starting {self._num_episodes} episodes, "
                  f"{self._episode_duration}s each")

    def consume_world_reset_request(self) -> bool:
        with self._lock:
            if self._world_reset_requested:
                self._world_reset_requested = False
                return True
            return False

    def stop_collection(self) -> None:
        with self._lock:
            self._collecting = False
            print("[AutoCollection] Collection stopped.")

    def reset_after_complete(self) -> None:
        with self._lock:
            self._collecting = False
            self._current_episode = 0

    def get_config(self) -> dict:
        with self._lock:
            return {
                "num_episodes": self._num_episodes,
                "episode_duration_sec": self._episode_duration,
                "arena_bounds": self._arena_bounds,
            }

    def update_config(self, config: dict) -> None:
        with self._lock:
            self._update_config_locked(config)

    def _update_config_locked(self, config: dict) -> None:
        if "num_episodes" in config:
            self._num_episodes = int(config["num_episodes"])
        dur = config.get("episode_duration") or config.get("episode_duration_sec")
        if dur is not None:
            self._episode_duration = float(dur)
        if "arena_bounds" in config:
            self._arena_bounds.update(config["arena_bounds"])
            self._wp_gen.bounds = self._arena_bounds

    def get_status(self) -> dict:
        with self._lock:
            ep_duration = (time.time() - self._episode_start_time
                           if self._episode_start_time and self._collecting else 0.0)
            return {
                "active": self._collecting,
                "current_episode": self._current_episode,
                "total_episodes": self._num_episodes,
                "episode_duration": round(ep_duration, 1),
                "current_waypoint": list(self._current_wp) if self._current_wp else None,
                "goals_reached": self._goals_reached,
            }

    def is_collecting(self) -> bool:
        with self._lock:
            return self._collecting

    def is_complete(self) -> bool:
        with self._lock:
            return not self._collecting and self._current_episode >= self._num_episodes


class SpotGaitDataCollector:
    """
    Autonomous gait data collection: single robot, fixed spawn position,
    teleports back to spawn after each episode.
    """

    def __init__(self, physics_dt: float, render_dt: float, config: dict):
        self._physics_dt = physics_dt
        self._config = config

        self._world = World(stage_units_in_meters=1.0,
                            physics_dt=physics_dt, rendering_dt=render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        prim = define_prim("/World/Warehouse", "Xform")
        asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
        prim.GetReferences().AddReference(asset_path)

        policy_path = os.path.join(BASE_DIR, "policies/spot_flat/models", "policy.pt")
        params_path = os.path.join(BASE_DIR, "policies/spot_flat/params", "env.yaml")
        usd_path    = os.path.join(BASE_DIR, "assets", "spot_sensors.usd")

        nav_policy_path = os.path.join(BASE_DIR, "policies/spot_nav/models", "policy.pt")
        self._nav_policy = SpotNavigationPolicy(nav_policy_path)
        print(f"[GaitCollector] Nav policy loaded. obs_dim={self._nav_policy.obs_dim}")

        self._nav_decimation = 100
        self._pos_thresh = config.get("pos_thresh", 0.5)
        self._yaw_thresh = config.get("yaw_thresh", 0.4)

        # Fixed spawn position — robot returns here after every episode
        spawn_x   = config.get("spawn_x",   -8.0)
        spawn_y   = config.get("spawn_y",    4.0)
        spawn_yaw = config.get("spawn_yaw",  0.0)
        self._spawn_pos = np.array([spawn_x, spawn_y, 0.8])
        self._spawn_yaw = spawn_yaw
        cy, sy = math.cos(spawn_yaw / 2), math.sin(spawn_yaw / 2)
        self._spawn_quat = np.array([cy, 0.0, 0.0, sy])

        self._spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            usd_path=usd_path,
            policy_path=policy_path,
            policy_params_path=params_path,
            position=self._spawn_pos.copy(),
            orientation=self._spawn_quat.copy(),
        )
        self._bridge = ContactSensorBridge("/World/Spot")
        self._foot_prims = None

        # Navigation state
        self._base_cmd     = np.zeros(3)
        self._nav_counter  = 0
        self._current_wp   = None
        self._goals_reached = 0
        self._ep_start_time: float = None
        self._step_buf: list = []
        self._waypoints_visited: list = []

        # Arena bounds for goal sampling
        self._arena_bounds = {
            "x_min": config.get("arena_x_min", -35.0),
            "x_max": config.get("arena_x_max",   4.0),
            "y_min": config.get("arena_y_min", -25.0),
            "y_max": config.get("arena_y_max",  28.0),
        }

        # State bridge & recorder
        self._state_manager = StateManager()
        self._recorder = GaitRecorder(
            save_dir=os.path.join(BASE_DIR, "data/gait_recordings")
        )

        self._wp_gen = RandomWaypointGenerator(
            arena_bounds=self._arena_bounds,
            min_distance=config.get("min_waypoint_dist", 2.0),
            max_distance=config.get("max_waypoint_dist", 8.0),
        )
        self._collection_mgr = AutoCollectionManager(
            num_episodes=config.get("num_episodes", 10),
            episode_duration_sec=config.get("episode_duration_sec", 60.0),
            arena_bounds=self._arena_bounds,
            waypoint_generator=self._wp_gen,
            recorder=self._recorder,
        )

        self.first_step = True
        self.needs_reset = False

    def setup(self) -> None:
        self._bridge.pre_reset_setup()

        self._appwindow = omni.appwindow.get_default_app_window()
        self._world.add_physics_callback("spot_gait_forward",
                                         callback_fn=self._on_physics_step)

        start_gait_dashboard_server(
            state_manager=self._state_manager,
            recorder=self._recorder,
            collection_manager=self._collection_mgr,
            port=self._config.get("dashboard_port", 8001),
        )

        if self._config.get("auto_start", True):
            self._collection_mgr.start_collection()

    def _initialize(self) -> None:
        self._spot.initialize()
        self._bridge.post_reset_setup(physics_dt=self._physics_dt)
        foot_paths = self._bridge.foot_prim_paths
        if foot_paths:
            self._foot_prims = [XFormPrim(p) for p in foot_paths]
            print(f"[GaitCollector] Foot prims: {foot_paths}")
        else:
            print(f"[GaitCollector] WARNING: foot prims not found.")

    def _reset_to_spawn(self) -> None:
        """Teleport robot back to fixed spawn position without world.reset()."""
        self._spot.robot.set_world_pose(self._spawn_pos.copy(), self._spawn_quat.copy())
        self._spot.robot.set_linear_velocity(np.zeros(3))
        self._spot.robot.set_angular_velocity(np.zeros(3))
        n_dof = len(self._spot.default_pos)
        self._spot.robot.set_joint_positions(self._spot.default_pos[:n_dof])
        self._spot.robot.set_joint_velocities(np.zeros(n_dof))
        self._base_cmd    = np.zeros(3)
        self._current_wp  = None
        self._goals_reached = 0
        self._step_buf    = []
        self._waypoints_visited = []
        self._ep_start_time = None

    def _flush_episode(self) -> None:
        if not self._step_buf or self._ep_start_time is None:
            return
        buf = {}
        for key in self._recorder.FIELDS:
            buf[key] = [s[key] for s in self._step_buf if key in s]
        self._recorder.write_buffered_episode(
            buffer=buf,
            start_time=self._ep_start_time,
            arena_bounds=self._arena_bounds,
            waypoints=self._waypoints_visited,
        )
        self._step_buf = []
        self._waypoints_visited = []
        self._ep_start_time = None
        self._goals_reached = 0

    def _get_robot_pose(self) -> tuple:
        pos_IB, q_IB = self._spot.robot.get_world_pose()
        w, x, y, z = q_IB[0], q_IB[1], q_IB[2], q_IB[3]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(pos_IB[0]), float(pos_IB[1]), yaw

    def _compute_pose_command(self, pos_IB, q_IB, wp: tuple) -> np.ndarray:
        wx, wy, wyaw = wp
        w, x, y, z = q_IB[0], q_IB[1], q_IB[2], q_IB[3]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        target_vec = np.array([wx, wy, pos_IB[2]]) - pos_IB
        cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
        R_yaw_inv = np.array([[cos_y, -sin_y, 0.0],
                               [sin_y,  cos_y, 0.0],
                               [0.0,    0.0,   1.0]])
        pos_command_b = R_yaw_inv @ target_vec
        heading_error = _wrap_to_pi(wyaw - yaw)
        pose_dim = self._nav_policy.obs_dim - 6
        if pose_dim == 4:
            return np.array([pos_command_b[0], pos_command_b[1], pos_command_b[2], heading_error])
        return np.array([pos_command_b[0], pos_command_b[1], heading_error])

    def _get_foot_positions(self) -> np.ndarray:
        if self._foot_prims is None:
            return np.zeros((4, 3), dtype=np.float32)
        positions = []
        for prim in self._foot_prims:
            try:
                pos, _ = prim.get_world_pose()
                positions.append(pos[:3])
            except Exception:
                positions.append(np.zeros(3))
        return np.array(positions, dtype=np.float32)

    def _collect_gait_state(self, pos_IB, q_IB) -> dict:
        lin_vel_I  = self._spot.robot.get_linear_velocity()
        ang_vel_I  = self._spot.robot.get_angular_velocity()
        joint_pos  = self._spot.robot.get_joint_positions()
        joint_vel  = self._spot.robot.get_joint_velocities()

        foot_positions   = self._get_foot_positions()
        contact_forces   = self._bridge.get_contact_forces()
        air_times        = self._bridge.get_air_times()
        foot_in_contact  = self._bridge.get_foot_in_contact()

        com_pos  = np.array(pos_IB[:3], dtype=np.float32)
        zmp      = compute_zmp(foot_positions, contact_forces)
        head_pos = com_pos.copy()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.T
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(48, dtype=np.float32)
        obs[:3]    = lin_vel_b
        obs[3:6]   = ang_vel_b
        obs[6:9]   = gravity_b
        obs[9:12]  = self._base_cmd
        obs[12:24] = joint_pos[:12] - self._spot.default_pos[:12]
        obs[24:36] = joint_vel[:12]
        obs[36:48] = self._spot._previous_action[:12]

        return {
            "timestamp":          time.time(),
            "obs":                obs.tolist(),
            "actions":            self._spot._previous_action[:12].tolist(),
            "commands":           self._base_cmd.tolist(),
            "body_pose":          np.concatenate([pos_IB[:3], q_IB]).tolist(),
            "body_lin_vel":       lin_vel_I[:3].tolist(),
            "body_ang_vel":       ang_vel_I[:3].tolist(),
            "com_position":       com_pos.tolist(),
            "zmp_position":       zmp.tolist(),
            "foot_positions":     foot_positions.tolist(),
            "foot_contact_forces": contact_forces.tolist(),
            "foot_air_time":      air_times.tolist(),
            "head_position":      head_pos.tolist(),
            "joint_pos":          joint_pos[:12].tolist(),
            "joint_vel":          joint_vel[:12].tolist(),
            "foot_contact":       foot_in_contact.tolist(),  # dashboard only, not in HDF5
        }

    def _on_physics_step(self, step_size) -> None:
        if self.first_step:
            self._initialize()
            self.first_step = False
            return

        if self.needs_reset or self._collection_mgr.consume_world_reset_request():
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
            return

        is_collecting = self._collection_mgr.is_collecting()

        self._bridge.update(step_size)

        # Nav policy runs at decimated rate
        if self._nav_counter % self._nav_decimation == 0:
            pos_IB, q_IB = self._spot.robot.get_world_pose()
            lin_vel_I = self._spot.robot.get_linear_velocity()
            R_BI = quat_to_rot_matrix(q_IB).T
            lin_vel_b = np.matmul(R_BI, lin_vel_I)
            gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
            current_pose = self._get_robot_pose()

            if is_collecting:
                now = time.time()

                # Start episode
                if self._ep_start_time is None:
                    goal = self._wp_gen.sample(current_pose)
                    self._current_wp = goal
                    self._ep_start_time = now
                    self._step_buf = []
                    self._waypoints_visited = []
                    self._goals_reached = 0
                    print(f"\n[Episode {self._collection_mgr._current_episode + 1}] "
                          f"Started → goal=({goal[0]:.2f},{goal[1]:.2f})")

                # Episode timeout
                elif now - self._ep_start_time >= self._collection_mgr._episode_duration:
                    ep_idx = self._collection_mgr._current_episode + 1
                    print(f"\n[Episode {ep_idx}] Done "
                          f"({now - self._ep_start_time:.1f}s), "
                          f"{self._goals_reached} goals")
                    self._flush_episode()
                    with self._collection_mgr._lock:
                        self._collection_mgr._current_episode += 1
                        done = (self._collection_mgr._current_episode >=
                                self._collection_mgr._num_episodes)
                        if done:
                            self._collection_mgr._collecting = False
                            print("\n[GaitCollector] All episodes collected! "
                                  "Idling — use dashboard to start new collection.")
                            self._collection_mgr.reset_after_complete()

                    if not done:
                        # Return to fixed spawn for next episode
                        self._reset_to_spawn()
                        goal = self._wp_gen.sample((self._spawn_pos[0], self._spawn_pos[1], 0.0))
                        self._current_wp = goal
                        self._ep_start_time = now

                # Navigate toward current waypoint
                if self._current_wp is not None and is_collecting:
                    wx, wy, wyaw = self._current_wp
                    dist = math.hypot(pos_IB[0] - wx, pos_IB[1] - wy)
                    yaw_curr = math.atan2(
                        2 * (q_IB[0] * q_IB[3] + q_IB[1] * q_IB[2]),
                        1 - 2 * (q_IB[2] ** 2 + q_IB[3] ** 2)
                    )
                    if (dist < self._pos_thresh and
                            abs(_wrap_to_pi(yaw_curr - wyaw)) < self._yaw_thresh):
                        self._goals_reached += 1
                        self._waypoints_visited.append(list(self._current_wp))
                        self._current_wp = self._wp_gen.sample(current_pose)

                    pose_cmd = self._compute_pose_command(pos_IB, q_IB, self._current_wp)
                    nav_obs = np.concatenate([lin_vel_b, gravity_b, pose_cmd])
                    raw_cmd = self._nav_policy.forward(nav_obs)
                    self._base_cmd = np.array([
                        np.clip(raw_cmd[0], -2.0, 3.0),
                        np.clip(raw_cmd[1], -1.5, 1.5),
                        np.clip(raw_cmd[2], -2.0, 2.0),
                    ])

                # Buffer gait state
                if self._ep_start_time is not None:
                    gait_state = self._collect_gait_state(pos_IB, q_IB)
                    self._step_buf.append(gait_state)
                    self._state_manager.push_state(gait_state)

            else:
                self._base_cmd = np.zeros(3)

        self._nav_counter += 1

        if is_collecting:
            self._spot.forward(step_size, self._base_cmd)

    def run(self) -> None:
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True

    def shutdown(self) -> None:
        self._flush_episode()
        self._recorder.close()


def main():
    parser = argparse.ArgumentParser(description="Spot Gait Data Collection")
    parser.add_argument("--num-episodes",     type=int,   default=10)
    parser.add_argument("--episode-duration", type=float, default=60.0)
    parser.add_argument("--spawn-x",   type=float, default=-8.0,
                        help="Fixed spawn X position (default: -8.0)")
    parser.add_argument("--spawn-y",   type=float, default=4.0,
                        help="Fixed spawn Y position (default: 4.0)")
    parser.add_argument("--spawn-yaw", type=float, default=0.0,
                        help="Fixed spawn yaw in radians (default: 0.0)")
    parser.add_argument("--arena-x-min", type=float, default=-35.0)
    parser.add_argument("--arena-x-max", type=float, default=4.0)
    parser.add_argument("--arena-y-min", type=float, default=-25.0)
    parser.add_argument("--arena-y-max", type=float, default=28.0)
    parser.add_argument("--min-goal-dist", type=float, default=2.0)
    parser.add_argument("--max-goal-dist", type=float, default=15.0)
    parser.add_argument("--pos-thresh",    type=float, default=0.5)
    parser.add_argument("--yaw-thresh",    type=float, default=0.4)
    parser.add_argument("--dashboard-port", type=int,  default=8001)
    parser.add_argument("--no-auto-start",  action="store_true")
    args, _ = parser.parse_known_args()

    config = {
        "num_episodes":        args.num_episodes,
        "episode_duration_sec": args.episode_duration,
        "spawn_x":             args.spawn_x,
        "spawn_y":             args.spawn_y,
        "spawn_yaw":           args.spawn_yaw,
        "arena_x_min":         args.arena_x_min,
        "arena_x_max":         args.arena_x_max,
        "arena_y_min":         args.arena_y_min,
        "arena_y_max":         args.arena_y_max,
        "min_waypoint_dist":   args.min_goal_dist,
        "max_waypoint_dist":   args.max_goal_dist,
        "pos_thresh":          args.pos_thresh,
        "yaw_thresh":          args.yaw_thresh,
        "dashboard_port":      args.dashboard_port,
        "auto_start":          not args.no_auto_start,
    }

    physics_dt = 0.002
    render_dt  = 1 / 60.0

    print(f"[GaitCollector] {args.num_episodes} episodes, spawn=({args.spawn_x},{args.spawn_y})")
    collector = SpotGaitDataCollector(physics_dt=physics_dt, render_dt=render_dt, config=config)

    simulation_app.update()
    collector._world.reset()
    simulation_app.update()
    collector.setup()
    # Second reset activates ContactSensor prims created in pre_reset_setup()
    collector._world.reset()
    simulation_app.update()

    print(f"[GaitCollector] Dashboard: http://localhost:{args.dashboard_port}")
    print(f"[GaitCollector] Running autonomous collection...")

    try:
        collector.run()
    except KeyboardInterrupt:
        print("\n[GaitCollector] Interrupted by user.")
    finally:
        collector.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
