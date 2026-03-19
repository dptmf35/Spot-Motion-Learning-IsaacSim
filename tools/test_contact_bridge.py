"""
Test: ContactSensorBridge with isaacsim.sensors.physics.ContactSensor.
Verifies that the ContactSensorBridge produces non-zero Fz after world.reset().
All output → /tmp/diag_output.txt  (Kit hijacks stdout)
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "applications"))
sys.path.insert(0, str(BASE_DIR / "dashboard"))

_log = open("/tmp/diag_output.txt", "w")
def log(msg=""):
    _log.write(str(msg) + "\n")
    _log.flush()

import carb
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path
from spot_policy import SpotFlatTerrainPolicy
from backend.contact_sensor_bridge import ContactSensorBridge, compute_zmp

PHYSICS_DT = 0.002
world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=1/60.0)

assets_root = get_assets_root_path()
prim = define_prim("/World/Warehouse", "Xform")
prim.GetReferences().AddReference(assets_root + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd")

spot = SpotFlatTerrainPolicy(
    prim_path="/World/Spot", name="Spot",
    usd_path=str(BASE_DIR / "assets/spot_sensors.usd"),
    policy_path=str(BASE_DIR / "policies/spot_flat/models/policy.pt"),
    policy_params_path=str(BASE_DIR / "policies/spot_flat/params/env.yaml"),
    position=np.array([-8.0, 4.0, 0.8]),
    orientation=np.array([0.7071, 0.0, 0.0, 0.7071]),
)

bridge = ContactSensorBridge("/World/Spot")

# --- pre_reset_setup: apply API + create sensor prims ---
simulation_app.update()
world.reset()
simulation_app.update()

bridge.pre_reset_setup()    # creates ContactSensor prims in USD stage
log(f"Foot prims found: {bridge.foot_prim_paths}")

# --- Second reset: activates ContactSensor prims ---
world.reset()
simulation_app.update()
spot.initialize()
simulation_app.update()

# --- post_reset_setup: sensors now active ---
bridge.post_reset_setup(physics_dt=PHYSICS_DT)
log(f"Bridge initialized: {bridge.is_initialized}")

# --- Run physics steps and verify forces ---
log()
log("=== Running 300 physics steps (robot walking) ===")
base_cmd = np.array([1.0, 0.0, 0.0])
for step in range(300):
    world.step(render=False)
    spot.forward(PHYSICS_DT, base_cmd)
    bridge.update(PHYSICS_DT)

forces = bridge.get_contact_forces()
contacts = bridge.get_foot_in_contact()
log(f"Contact forces (FL/FR/HL/HR):")
log(f"  FL Fz={forces[0,2]:.1f}  FR Fz={forces[1,2]:.1f}  HL Fz={forces[2,2]:.1f}  HR Fz={forces[3,2]:.1f}")
log(f"  In contact: {contacts}")
log(f"  Any contact: {contacts.any()}")

fz = forces[:, 2]
if fz.max() > 1.0:
    log()
    log("[PASS] Non-zero contact forces detected!")
else:
    log()
    log("[FAIL] All forces are still zero!")

log()
log("=== DONE ===")
_log.close()
simulation_app.close()
