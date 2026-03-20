# pylint: disable=no-member
"""
DualDeviceNav Human Play - Generic Model

This script allows you to manually control dual devices in any processed VMR model.

Usage:
    python dual_human_play_general.py --model_name 0011
    python dual_human_play_general.py --model_name 0011_H_AO_H
    
    Or in Docker:
    docker run ... python3 .../dual_human_play_general.py --model_name 0011

Controls:
    - Arrow Keys: Control device movements
    - ENTER: Reset episode
    - ESC: Exit

Make sure you have run the processing script first to generate:
    <vmr_root>/<model_name>/dualdevicenav_format/
"""

from time import perf_counter
import pygame
import os
import sys
import argparse
from pathlib import Path

from eve_bench.dualdevicenav import DualDeviceNavCustom, load_branches
from eve.visualisation import SofaPygame
from eve.util.userinput.instrumentaction import KeyboardTwoDevice
from eve.util.userinput.visumanipulator import VisuManipulator


def find_model_folder(vmr_root: str, model_identifier: str) -> tuple:
    """
    Find the model folder given a model identifier.
    
    Args:
        vmr_root: Root directory containing VMR models (e.g., "D:\\vmr\\vmr" or "/vmr_host/vmr")
        model_identifier: Model name or number (e.g., "0011" or "0011_H_AO_H")
    
    Returns:
        tuple: (full_model_name, model_folder_path, dualdevicenav_format_path)
    
    Raises:
        FileNotFoundError: If model not found
        ValueError: If multiple matches found
    """
    vmr_root_path = Path(vmr_root)
    
    if not vmr_root_path.exists():
        raise FileNotFoundError(f"VMR root directory not found: {vmr_root}")
    
    # If model_identifier looks like a full name (contains underscores), use it directly
    if '_' in model_identifier:
        full_model_name = model_identifier
        model_folder = vmr_root_path / full_model_name
    else:
        # Search for folders starting with the number
        matching_folders = []
        for folder in vmr_root_path.iterdir():
            if folder.is_dir() and folder.name.startswith(model_identifier + '_'):
                matching_folders.append(folder)
        
        if len(matching_folders) == 0:
            raise FileNotFoundError(
                f"No model found starting with '{model_identifier}' in {vmr_root}\n"
                f"Available models: {sorted([f.name for f in vmr_root_path.iterdir() if f.is_dir()])[:10]}..."
            )
        elif len(matching_folders) > 1:
            print(f"WARNING: Multiple models found starting with '{model_identifier}':")
            for folder in matching_folders:
                print(f"  - {folder.name}")
            print(f"Using first match: {matching_folders[0].name}")
            full_model_name = matching_folders[0].name
            model_folder = matching_folders[0]
        else:
            full_model_name = matching_folders[0].name
            model_folder = matching_folders[0]
    
    dualdevicenav_format = model_folder / "dualdevicenav_format"
    
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
    
    if not dualdevicenav_format.exists():
        raise FileNotFoundError(
            f"Processed data not found: {dualdevicenav_format}\n"
            f"Please run the processing script first:\n"
            f"  vmr_processing_tools\\run_dualdevicenav.bat {model_folder}"
        )
    
    return full_model_name, str(model_folder), str(dualdevicenav_format)


def main():
    parser = argparse.ArgumentParser(
        description="DualDeviceNav Human Play - Generic Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using model number
  python dual_human_play_general.py --model_name 0011
  
  # Using full model name
  python dual_human_play_general.py --model_name 0011_H_AO_H
  
  # In Docker
  docker run ... python3 .../dual_human_play_general.py --model_name 0011
        """
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model identifier: number (e.g., '0011') or full name (e.g., '0011_H_AO_H')"
    )
    
    parser.add_argument(
        "--vmr_root",
        type=str,
        default=None,
        help="Root directory containing VMR models (auto-detects Docker vs Windows)"
    )
    
    parser.add_argument(
        "--insertion_point",
        type=float,
        nargs=3,
        default=None,
        metavar=('X', 'Y', 'Z'),
        help="Custom insertion point [x, y, z] (default: auto-detect from branch 0)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect VMR root (Docker vs Windows)
    if args.vmr_root:
        vmr_root = args.vmr_root
    elif os.path.exists("/vmr_host/vmr"):
        vmr_root = "/vmr_host/vmr"
    else:
        vmr_root = r"D:\vmr\vmr"
    
    print("="*80)
    print("DualDeviceNav Human Play - Generic Model")
    print("="*80)
    print(f"Model identifier: {args.model_name}")
    print(f"VMR root: {vmr_root}")
    print("="*80)
    print()
    
    # Find model folder
    try:
        full_model_name, model_folder, dualdevicenav_format = find_model_folder(
            vmr_root, args.model_name
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"Found model: {full_model_name}")
    print(f"Model folder: {model_folder}")
    print(f"Processed data: {dualdevicenav_format}")
    print()
    
    # Load branches to get the correct insertion point
    try:
        branches = load_branches(os.path.join(dualdevicenav_format, "Centrelines"))
    except Exception as e:
        print(f"ERROR: Failed to load centerlines: {e}")
        sys.exit(1)
    
    if len(branches) == 0:
        print("ERROR: No branches found in centerlines!")
        sys.exit(1)
    
    # Determine insertion point
    if args.insertion_point:
        insertion_point = list(args.insertion_point)
        print(f"Using custom insertion point: {insertion_point}")
    else:
        # Use the LAST point of branch 0 as insertion point (correct for most models)
        insertion_point = branches[0].coordinates[-1].tolist()
        print(f"Using auto-detected insertion point (last point of branch 0): {insertion_point}")
        print(f"  Alternative (first point): {branches[0].coordinates[0].tolist()}")
        print(f"  If insertion point is wrong, use --insertion_point X Y Z")
    
    # Configuration for no-rotation preprocessing
    # - Data has NO rotation in OBJ/JSON (only scaling cm->mm)
    # - Branches get (y,-z,-x) transformation when loaded (automatic)
    # - Mesh needs [90,-90,0] rotation to match SOFA coordinate system
    # - rotate_branches/rotate_ip always False (branches already transformed)
    print()
    print("Creating intervention with standard settings:")
    print("  rotation_yzx_deg=[90, -90, 0]  # Rotate mesh to match SOFA coordinates")
    print("  fluoroscopy_rot_zx=[20, 5]      # Standard fluoroscopy angles")
    print()
    
    try:
        intervention = DualDeviceNavCustom(
            mesh_folder=dualdevicenav_format,
            model_name=full_model_name,
            insertion_point=insertion_point,
            rotation_yzx_deg=[90, -90, 0],  # Rotate mesh to match SOFA coordinates
            fluoroscopy_rot_zx=[20, 5]       # Standard fluoroscopy angles
        )
    except Exception as e:
        print(f"ERROR: Failed to create intervention: {e}")
        sys.exit(1)
    
    visu = SofaPygame(intervention)
    instrumentaction = KeyboardTwoDevice()
    visumanipulator = VisuManipulator(visu)
    
    n_steps = 0
    r_cum = 0.0
    
    intervention.reset()
    visu.reset()
    
    print("\n" + "="*80)
    print(f"DualDeviceNav - Model {full_model_name}")
    print("="*80)
    print("Controls:")
    print("  Arrow Keys: Control device movements")
    print("  ENTER: Reset episode")
    print("  ESC: Exit")
    print("="*80 + "\n")
    
    try:
        while True:
            start = perf_counter()
            
            pygame.event.get()
            keys_pressed = pygame.key.get_pressed()
            action = instrumentaction.get_action()
            visumanipulator.step()
            intervention.step(action=action)
            image = visu.render()
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
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Exiting...")
    finally:
        intervention.close()
        visu.close()
        print("\nClosed intervention and visualization.")


if __name__ == "__main__":
    main()

