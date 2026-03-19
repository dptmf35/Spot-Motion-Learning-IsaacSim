"""
ContactSensorBridge — contact force collection for standalone Isaac Sim apps.

Uses isaacsim.sensors.physics.ContactSensor (Isaac Sim 5.0 native API).
ContactSensor prims are created in pre_reset_setup(); a subsequent world.reset()
activates them so they report contact data from the first physics step.

Foot prim paths are auto-discovered (fl_foot / fr_foot / hl_foot / hr_foot).
PhysxContactReportAPI is applied to rigid bodies before world.reset().

Usage:
    bridge = ContactSensorBridge("/World/Spot")
    bridge.pre_reset_setup()   # BEFORE world.reset() — creates sensor prims
    world.reset()              # activates sensor prims
    bridge.post_reset_setup()  # after world.reset() + spot.initialize()

    # In physics step callback:
    bridge.update(dt)
    forces   = bridge.get_contact_forces()   # (4, 3)  FL/FR/HL/HR (Fz = contact force mag)
    air_time = bridge.get_air_times()        # (4,)
"""
import numpy as np

# Actual foot prim names in Spot USD (lowercase; hind legs = hl/hr, not rl/rr)
FOOT_ORDER = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]


def compute_zmp(foot_positions: np.ndarray, contact_forces: np.ndarray,
                threshold: float = 1.0) -> np.ndarray:
    """
    ZMP from foot contact forces (pressure-centre method).
    foot_positions : (4, 3) world positions  FL/FR/HL/HR
    contact_forces : (4, 3) GRF [Fx, Fy, Fz] in world frame
    Returns        : (2,) ZMP [x, y], or [nan, nan] if no contact
    """
    fz = contact_forces[:, 2]
    mask = fz > threshold
    if not mask.any():
        return np.array([np.nan, np.nan], dtype=np.float32)
    fz_v = fz[mask]
    total = fz_v.sum()
    zmp_x = (fz_v * foot_positions[mask, 0]).sum() / total
    zmp_y = (fz_v * foot_positions[mask, 1]).sum() / total
    return np.array([zmp_x, zmp_y], dtype=np.float32)


def _find_foot_prims(robot_prim_path: str) -> list:
    """
    Traverse USD stage to find all foot links under robot_prim_path.
    Returns list of prim paths in FL/FR/HL/HR order.
    """
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return []

        robot_prefix = robot_prim_path.rstrip("/")
        found_ordered = {}   # foot_order_key → path
        found_extra = []     # other foot prims not matching FOOT_ORDER

        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            if not path_str.startswith(robot_prefix + "/"):
                continue
            name = prim.GetName()
            if "foot" not in name.lower():
                continue
            matched = False
            for key in FOOT_ORDER:
                if name.lower() == key.lower():
                    found_ordered[key] = path_str
                    matched = True
                    break
            if not matched:
                found_extra.append(path_str)

        result = [found_ordered[k] for k in FOOT_ORDER if k in found_ordered]
        result += found_extra

        if result:
            print(f"[ContactBridge] Found foot prims ({len(result)}): {result}")
        else:
            print(f"[ContactBridge] WARNING: No foot prims found under {robot_prefix}")
        return result
    except Exception as e:
        print(f"[ContactBridge] USD traversal error: {e}")
        return []


class ContactSensorBridge:
    """
    Contact force bridge using isaacsim.sensors.physics.ContactSensor.
    Requires Isaac Sim 5.0+. No IsaacLab dependency.
    """

    def __init__(self, robot_prim_path: str, force_threshold: float = 1.0):
        self._robot_prim_path = robot_prim_path
        self._force_threshold = force_threshold

        self._contact_sensors = []          # list of ContactSensor objects
        self._foot_prim_paths: list = []
        self._num_feet = 4
        self._physics_dt = 0.002
        self._initialized = False

        self._contact_forces = np.zeros((4, 3), dtype=np.float32)
        self._air_times = np.zeros(4, dtype=np.float32)
        self._debug_step = 0
        # Skip first N update() calls to allow sensor warmup
        self._post_init_steps = 0
        self._POST_INIT_WARMUP = 5

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def pre_reset_setup(self) -> None:
        """
        Call BEFORE world.reset().
        1. Applies PhysxContactReportAPI to robot rigid bodies.
        2. Discovers foot prim paths.
        3. Creates ContactSensor prims on each foot.

        world.reset() must be called AFTER this to activate the sensor prims.
        """
        # 1. Apply PhysxContactReportAPI to all rigid bodies under robot
        applied = 0
        try:
            import omni.usd
            from pxr import UsdPhysics, PhysxSchema
            stage = omni.usd.get_context().get_stage()
            robot_prefix = self._robot_prim_path.rstrip("/")
            for prim in stage.Traverse():
                path_str = str(prim.GetPath())
                if not path_str.startswith(robot_prefix + "/"):
                    continue
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                        PhysxSchema.PhysxContactReportAPI.Apply(prim)
                        applied += 1
            print(f"[ContactBridge] PhysxContactReportAPI applied to {applied} rigid bodies")
        except Exception as e:
            print(f"[ContactBridge] WARNING: Could not apply PhysxContactReportAPI: {e}")

        # 2. Discover foot prims (only if not already done)
        if not self._foot_prim_paths:
            self._foot_prim_paths = _find_foot_prims(self._robot_prim_path)
            self._num_feet = len(self._foot_prim_paths) if self._foot_prim_paths else 4
            self._contact_forces = np.zeros((self._num_feet, 3), dtype=np.float32)
            self._air_times = np.zeros(self._num_feet, dtype=np.float32)

        # 3. Create ContactSensor prims (only if not already created)
        if not self._contact_sensors and self._foot_prim_paths:
            self._create_contact_sensors()

    def post_reset_setup(self, physics_dt: float = 0.002) -> None:
        """
        Call AFTER world.reset() + spot.initialize().
        ContactSensor prims are now registered with PhysX.
        """
        self._physics_dt = physics_dt
        self._post_init_steps = 0
        self._debug_step = 0
        self._contact_forces = np.zeros((self._num_feet, 3), dtype=np.float32)
        self._air_times = np.zeros(self._num_feet, dtype=np.float32)

        if self._contact_sensors:
            self._initialized = True
            print(f"[ContactBridge] Ready: {len(self._contact_sensors)} ContactSensors active")
        else:
            self._initialized = False
            print("[ContactBridge] WARNING: No ContactSensors — contact data will be zeros")

    def update(self, dt: float) -> None:
        """Call every physics step to refresh contact force data."""
        if not self._initialized:
            self._debug_step += 1
            in_contact = self._contact_forces[:, 2] > self._force_threshold
            self._air_times[~in_contact] += dt
            self._air_times[in_contact] = 0.0
            return

        # Warmup: skip first few calls while PhysX contact registry stabilises
        if self._post_init_steps < self._POST_INIT_WARMUP:
            self._post_init_steps += 1
            self._debug_step += 1
            return

        try:
            for i, cs in enumerate(self._contact_sensors):
                frame = cs.get_current_frame()
                # ContactSensor returns scalar force magnitude; store as Fz
                # (on flat terrain, contact force is predominantly vertical)
                force = float(frame.get("force", 0.0))
                self._contact_forces[i, 0] = 0.0
                self._contact_forces[i, 1] = 0.0
                self._contact_forces[i, 2] = force
        except Exception as e:
            if self._debug_step % 500 == 0:
                print(f"[ContactBridge] update() error: {e}")

        if self._debug_step % 500 == 0:
            fz = self._contact_forces[:, 2]
            in_contact = fz > self._force_threshold
            print(f"[ContactBridge] step={self._debug_step} Fz={fz.round(1)} contact={in_contact}")
        self._debug_step += 1

        in_contact = self._contact_forces[:, 2] > self._force_threshold
        self._air_times[~in_contact] += dt
        self._air_times[in_contact] = 0.0

    # ------------------------------------------------------------------ #
    #  Public accessors
    # ------------------------------------------------------------------ #

    def get_contact_forces(self) -> np.ndarray:
        """(4, 3) GRF [Fx, Fy, Fz] world frame. Order: FL/FR/HL/HR."""
        return self._contact_forces.copy()

    def get_air_times(self) -> np.ndarray:
        """(4,) seconds airborne per foot."""
        return self._air_times.copy()

    def get_foot_in_contact(self) -> np.ndarray:
        """(4,) bool: True when foot Fz > threshold."""
        return self._contact_forces[:, 2] > self._force_threshold

    @property
    def foot_prim_paths(self) -> list:
        return list(self._foot_prim_paths)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _create_contact_sensors(self) -> None:
        """
        Create isaacsim.sensors.physics.ContactSensor for each foot prim.
        Sensor prims are placed at <foot_path>/contact_sensor.
        They become active after the next world.reset().
        """
        try:
            from isaacsim.sensors.physics import ContactSensor
            self._contact_sensors = []
            for fp in self._foot_prim_paths:
                foot_name = fp.split('/')[-1]
                cs = ContactSensor(
                    prim_path=fp + "/contact_sensor",
                    name=f"cs_{foot_name}",
                    min_threshold=0.0,
                    max_threshold=1e10,
                    radius=-1,
                )
                self._contact_sensors.append(cs)
            print(f"[ContactBridge] Created {len(self._contact_sensors)} ContactSensors "
                  f"(will activate after next world.reset())")
        except Exception as e:
            print(f"[ContactBridge] ContactSensor creation failed: {e}")
            self._contact_sensors = []
