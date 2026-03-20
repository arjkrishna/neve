# pylint: disable=no-member
"""
DualDeviceNav Human Play - Model 0011

This script allows you to manually control dual devices in the 0011_H_AO_H model
(newly processed from VMR data) instead of the default model 0105.

Controls:
    - Arrow Keys: Control device movements
    - ENTER: Reset episode
    - ESC: Exit

Make sure you have run the processing script first to generate:
    D:/vmr/vmr/0011_H_AO_H/dualdevicenav_format/
"""

from time import perf_counter
import pygame
import os
from pathlib import Path

from eve_bench.dualdevicenav import DualDeviceNavCustom
from eve.visualisation import SofaPygame
from eve.util.userinput.instrumentaction import KeyboardTwoDevice
from eve.util.userinput.visumanipulator import VisuManipulator

# Path to the newly processed model 0011 data
# Check if running in Docker (mounted at /vmr_host) or Windows
if os.path.exists("/vmr_host/vmr/0011_H_AO_H/dualdevicenav_format"):
    MODEL_0011_PATH = "/vmr_host/vmr/0011_H_AO_H/dualdevicenav_format"
else:
    MODEL_0011_PATH = r"D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format"

# Check if the data exists
if not os.path.exists(MODEL_0011_PATH):
    print(f"ERROR: Model 0011 data not found at: {MODEL_0011_PATH}")
    print("Please run the processing script first:")
    print("  vmr_processing_tools\\run_dualdevicenav.bat")
    exit(1)

# Create intervention with custom model 0011 data
# Preprocessing: Data has NO rotation applied (only scaling cm->mm)
# stEVE applies rotation via rotation_yzx_deg=[90, -90, 0]
# 
# rotation_yzx_deg: Rotates the 3D mesh, centerlines, insertion point
# fluoroscopy_rot_zx: Rotates the 2D camera view angle
# Load branches to get the correct insertion point
from eve_bench.dualdevicenav import load_branches
import os
branches = load_branches(os.path.join(MODEL_0011_PATH, "Centrelines"))

# Use the LAST point of branch 0 as insertion point (correct for this model)
insertion_point = branches[0].coordinates[-1].tolist()
print(f"Using insertion point (last point of branch 0): {insertion_point}")

# Configuration for no-rotation preprocessing
# - Data has NO rotation in OBJ/JSON (only scaling cm->mm)
# - Branches get (y,-z,-x) transformation when loaded (automatic)
# - Mesh needs [90,-90,0] rotation to match SOFA coordinate system
# - rotate_branches/rotate_ip always False (branches already transformed)
intervention = DualDeviceNavCustom(
    mesh_folder=MODEL_0011_PATH,
    model_name="0011_H_AO_H",
    insertion_point=insertion_point,
    rotation_yzx_deg=[90, -90, 0],  # Rotate mesh to match SOFA coordinates
    fluoroscopy_rot_zx=[20, 5]       # Standard fluoroscopy angles
)

visu = SofaPygame(intervention)

instrumentaction = KeyboardTwoDevice()
visumanipulator = VisuManipulator(visu)

n_steps = 0
r_cum = 0.0

intervention.reset()
visu.reset()

print("\n" + "="*80)
print("DualDeviceNav - Model 0011_H_AO_H")
print("="*80)
print("Controls:")
print("  Arrow Keys: Control device movements")
print("  ENTER: Reset episode")
print("  ESC: Exit")
print("="*80 + "\n")

while True:
    start = perf_counter()

    pygame.event.get()
    keys_pressed = pygame.key.get_pressed()
    action = instrumentaction.get_action()
    visumanipulator.step()
    intervention.step(action=action)
    image = visu.render()
    # plt.imshow(image)
    # plt.show()
    n_steps += 1

    print(f"{n_steps=}")

    if keys_pressed[pygame.K_RETURN]:
        intervention.reset()
        intervention.reset_devices()
        visu.reset()
        n_steps = 0

    print(f"FPS: {1/(perf_counter()-start)}")

    if keys_pressed[pygame.K_ESCAPE]:
        break
        
intervention.close()
visu.close()

