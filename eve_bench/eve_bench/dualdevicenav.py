import os
import json
from pathlib import Path
import numpy as np
from typing import Tuple, List
import eve
from eve.intervention.vesseltree.util.branch import (
    Branch,
    BranchWithRadii,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data/dualdevicenav"


class DualDeviceNav(eve.intervention.MonoPlaneStatic):
    def __init__(
        self,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
    ) -> None:

        mesh = os.path.join(DATA_DIR, "vessel_architecture_collision.obj")
        visu_mesh = os.path.join(DATA_DIR, "vessel_architecture_visual.obj")

        centerline_folder_path = os.path.join(DATA_DIR, "Centrelines_comb")
        branches = load_branches(centerline_folder_path)

        insertion = [65.0, -5.0, 35.0]

        vessel_tree = eve.intervention.vesseltree.FromMesh(
            mesh,
            insertion,
            [-1.0, 0.0, 1.0],
            branch_list=branches,
            rotation_yzx_deg=[90, -90, 0],
            scaling_xyz=[1.0, 1.0, 1.0],
            rotate_branches=False,
            rotate_ip=False,
            visu_mesh=visu_mesh,
        )

        device1 = eve.intervention.device.JShaped(
            name="mic_guide",
            length=900,
            velocity_limit=(30, 1.5),
            visu_edges_per_mm=0.5,
            tip_outer_diameter=0.36,
            straight_outer_diameter=0.36,
            tip_inner_diameter=0,
            straight_inner_diameter=0.36,
            mass_density_tip=0.000005,
            mass_density_straight=0.000005,
            young_modulus_tip=1e3,
            young_modulus_straight=1e3,
            beams_per_mm_straight=0.6,
        )

        device2 = eve.intervention.device.JShaped(
            name="mic_cath",
            length=900,
            velocity_limit=(30, 1.5),
            visu_edges_per_mm=0.5,
            tip_outer_diameter=0.6,
            straight_outer_diameter=0.7,
            tip_inner_diameter=0.57,
            straight_inner_diameter=0.57,
            color=(1.0, 0.0, 0.0),
            mass_density_tip=0.000005,
            mass_density_straight=0.000005,
            young_modulus_tip=1e3,
            young_modulus_straight=1e3,
            beams_per_mm_straight=0.6,
        )

        simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)

        fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
            simulation=simulation,
            vessel_tree=vessel_tree,
            image_frequency=7.5,
            image_rot_zx=[20, 5],
        )

        target = eve.intervention.target.CenterlineRandom(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=5,
            branches=[
                "Centerline curve - LCCA.mrk",
                "Centerline curve - LVA.mrk",
                "Centerline curve - RCCA.mrk",
                "Centerline curve - RVA.mrk",
            ],
        )

        super().__init__(
            vessel_tree,
            [device1, device2],
            simulation,
            fluoroscopy,
            target,
            stop_device_at_tree_end,
            normalize_action,
            velocity_limit_low=np.array([[-10.0, -1.5], [-10.0, -1.5]]),
        )


def load_points_from_json(json_file_path: str) -> Tuple[Branch, List[float]]:
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    points = []
    radii = []
    for markup in data["markups"]:
        if markup["type"] == "Curve":
            control_points = markup["controlPoints"]
            for point in control_points:
                position = point["position"]
                x = float(position[0])
                y = float(position[1])
                z = float(position[2])
                points.append((y, -z, -x))  # Append as a tuple instead of a list

            if "measurements" in markup:
                measurements = markup["measurements"]
                for measurement in measurements:
                    if measurement["name"] == "Radius":
                        radii.extend(measurement["controlPointValues"])

    points = np.array(points, dtype=np.float32)
    filename = os.path.splitext(os.path.basename(json_file_path))[0]

    radii = np.array(radii, dtype=np.float32)
    branch = BranchWithRadii(name=filename, coordinates=points, radii=radii)

    return branch


def load_branches(folder_path: str) -> list:
    """
    Load branches from JSON files, sorted by branch number.
    
    Branch 0 (main trunk) should be "Centerline curve - X.mrk.json"
    Branch 1-N should be "Centerline curve (1).mrk.json", etc.
    """
    # Get all matching files
    files = [f for f in os.listdir(folder_path) 
             if f.startswith("Centerline curve ") and f.endswith(".json")]
    
    # Sort by branch number (branch 0 = file without parentheses)
    def get_branch_number(filename):
        if '(' not in filename:
            return 0  # "Centerline curve - X.mrk.json" = branch 0
        # Extract number from "Centerline curve (N).mrk.json"
        import re
        match = re.search(r'\((\d+)\)', filename)
        return int(match.group(1)) if match else 999
    
    files.sort(key=get_branch_number)
    
    # Load branches in sorted order
    centerlines = []
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        centerline = load_points_from_json(file_path)
        centerlines.append(centerline)
    
    return centerlines


class DualDeviceNavCustom(eve.intervention.MonoPlaneStatic):
    """
    DualDeviceNav with custom model data path.
    
    This allows using newly processed VMR models instead of the default model 0105.
    
    IMPORTANT: Designed for data created with create_dualdevicenav_format.py
    - OBJ/JSON data has NO rotation applied (only scaling cm->mm)
    - All rotation is handled by stEVE via rotation_yzx_deg
    - Uses rotate_branches=True and rotate_ip=True
    
    Args:
        mesh_folder: Path to folder containing:
            - vessel_architecture_collision.obj (or {model_name}_collision.obj)
            - vessel_architecture_visual.obj (or {model_name}_visual.obj)
            - Centrelines/ folder with .mrk.json files
        model_name: Name of the model (for display/logging)
        insertion_point: Custom insertion point (default auto-detects from first branch)
        rotation_yzx_deg: Mesh rotation [Y, Z, X] in degrees (default [90, -90, 0])
            - Rotates the 3D mesh, centerlines, and insertion point
            - Data should have NO rotation applied during preprocessing
        fluoroscopy_rot_zx: Camera view rotation [Z, X] in degrees (default [20, 5])
            - Controls the 2D camera view angle only
            - Does NOT affect 3D geometry or centerlines
        stop_device_at_tree_end: Whether to stop device at vessel tree end
        normalize_action: Whether to normalize actions
    """
    
    def __init__(
        self,
        mesh_folder: str,
        model_name: str = "custom",
        insertion_point: list = None,
        rotation_yzx_deg: list = None,
        fluoroscopy_rot_zx: list = None,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
    ) -> None:

        # Try default names first, then try model-specific names
        mesh = os.path.join(mesh_folder, "vessel_architecture_collision.obj")
        visu_mesh = os.path.join(mesh_folder, "vessel_architecture_visual.obj")
        
        if not os.path.exists(mesh):
            # Try model-specific naming: {model_name}_collision.obj
            mesh = os.path.join(mesh_folder, f"{model_name}_collision.obj")
        
        if not os.path.exists(visu_mesh):
            # Try model-specific naming: {model_name}_visual.obj
            visu_mesh = os.path.join(mesh_folder, f"{model_name}_visual.obj")

        # Check if files exist
        if not os.path.exists(mesh):
            raise FileNotFoundError(f"Collision mesh not found: {mesh}")
        if not os.path.exists(visu_mesh):
            raise FileNotFoundError(f"Visual mesh not found: {visu_mesh}")

        centerline_folder_path = os.path.join(mesh_folder, "Centrelines")
        if not os.path.exists(centerline_folder_path):
            raise FileNotFoundError(f"Centerlines folder not found: {centerline_folder_path}")
            
        branches = load_branches(centerline_folder_path)
        
        if len(branches) == 0:
            raise ValueError(f"No centerlines found in {centerline_folder_path}")
        
        # Use custom insertion point or derive from first branch
        if insertion_point is None:
            # Try both ends of the first branch to determine which is the insertion point
            # The insertion point should be at the aortic root (usually the start)
            # But VMTK might save branches in reverse, so check both ends
            first_point = branches[0].coordinates[0]
            last_point = branches[0].coordinates[-1]
            
            # For now, use the first point (aortic root should be at the start)
            # If this is wrong, the user can provide insertion_point manually
            insertion = first_point.tolist()
            print(f"Using auto-detected insertion point (first point of branch 0): {insertion}")
            print(f"  Alternative (last point): {last_point.tolist()}")
            print(f"  If insertion point is wrong, provide insertion_point parameter manually")
        else:
            insertion = insertion_point

        # Use custom rotation or default
        # Data should have NO rotation applied during preprocessing (create_dualdevicenav_format.py)
        # Mesh needs [90, -90, 0] to align with SOFA coordinate system
        if rotation_yzx_deg is None:
            rotation_yzx_deg = [90, -90, 0]  # Default: rotate mesh to match SOFA coordinates
        
        # IMPORTANT: Branches already have (y,-z,-x) transformation applied by load_branches()
        # So we should NEVER rotate them again, regardless of rotation_yzx_deg
        # The mesh rotation is separate and only affects the OBJ geometry
        vessel_tree = eve.intervention.vesseltree.FromMesh(
            mesh,
            insertion,
            [-1.0, 0.0, 1.0],
            branch_list=branches,
            rotation_yzx_deg=rotation_yzx_deg,
            scaling_xyz=[1.0, 1.0, 1.0],
            rotate_branches=False,  # Always False - branches already transformed by load_branches()
            rotate_ip=False,        # Always False - insertion point already in correct coordinate system
            visu_mesh=visu_mesh,
        )

        device1 = eve.intervention.device.JShaped(
            name="mic_guide",
            length=900,
            velocity_limit=(30, 1.5),
            visu_edges_per_mm=0.5,
            tip_outer_diameter=0.36,
            straight_outer_diameter=0.36,
            tip_inner_diameter=0,
            straight_inner_diameter=0.36,
            mass_density_tip=0.000005,
            mass_density_straight=0.000005,
            young_modulus_tip=1e3,
            young_modulus_straight=1e3,
            beams_per_mm_straight=0.6,
        )

        device2 = eve.intervention.device.JShaped(
            name="mic_cath",
            length=900,
            velocity_limit=(30, 1.5),
            visu_edges_per_mm=0.5,
            tip_outer_diameter=0.6,
            straight_outer_diameter=0.7,
            tip_inner_diameter=0.57,
            straight_inner_diameter=0.57,
            color=(1.0, 0.0, 0.0),
            mass_density_tip=0.000005,
            mass_density_straight=0.000005,
            young_modulus_tip=1e3,
            young_modulus_straight=1e3,
            beams_per_mm_straight=0.6,
        )

        simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)

        # Use custom fluoroscopy rotation or default
        # This controls the 2D view angle (what you see in the pygame window)
        if fluoroscopy_rot_zx is None:
            fluoroscopy_rot_zx = [20, 5]  # Default from original DualDeviceNav
        
        fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
            simulation=simulation,
            vessel_tree=vessel_tree,
            image_frequency=7.5,
            image_rot_zx=fluoroscopy_rot_zx,
        )

        # Auto-detect target branches (use all branches except the main one)
        # You can customize this based on your model's anatomy
        target_branches = [branch.name for branch in branches[1:]]  # Skip first (usually main trunk)
        
        if len(target_branches) == 0:
            print("WARNING: No target branches found, using all branches")
            target_branches = [branch.name for branch in branches]
        
        print(f"Target branches for model {model_name}: {target_branches}")

        target = eve.intervention.target.CenterlineRandom(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=5,
            branches=target_branches,
        )

        super().__init__(
            vessel_tree,
            [device1, device2],
            simulation,
            fluoroscopy,
            target,
            stop_device_at_tree_end,
            normalize_action,
            velocity_limit_low=np.array([[-10.0, -1.5], [-10.0, -1.5]]),
        )