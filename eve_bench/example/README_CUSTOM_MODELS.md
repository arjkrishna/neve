# Using Custom VMR Models with DualDeviceNav

This guide explains how to use newly processed VMR models with the DualDeviceNav environment.

## Quick Start

### 1. Process VMR Data

First, process a VMR model using the processing tools:

```bash
# Using the batch file (easiest)
cd D:\stEVE_training
vmr_processing_tools\run_dualdevicenav.bat

# Or using Python directly (requires vmtk_env conda environment)
conda activate vmtk_env
python vmr_processing_tools\create_dualdevicenav_format.py D:\vmr\vmr\0011_H_AO_H
```

This will create:
```
D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format\
├── vessel_architecture_collision.obj
├── vessel_architecture_visual.obj
└── Centrelines\
    ├── Centerline curve - 0011_H_AO_H.mrk.json
    ├── Centerline curve (1).mrk.json
    ├── Centerline curve (2).mrk.json
    └── ...
```

### 2. Run the Play Script

```bash
cd D:\stEVE_training
python eve_bench\example\dual_human_play_0011.py
```

## Available Scripts

### `dual_human_play_0011.py`
Interactive manual control of dual devices in model 0011.

**Controls:**
- Arrow Keys: Control device movements
- ENTER: Reset episode
- ESC: Exit

## Creating Scripts for Other Models

To use a different VMR model (e.g., 0015_H_AO_COA):

1. **Process the model:**
   ```bash
   python vmr_processing_tools\create_dualdevicenav_format.py D:\vmr\vmr\0015_H_AO_COA
   ```

2. **Create a new play script:**
   ```python
   from eve_bench.dualdevicenav import DualDeviceNavCustom
   from eve.visualisation import SofaPygame
   
   MODEL_PATH = r"D:\vmr\vmr\0015_H_AO_COA\dualdevicenav_format"
   
   intervention = DualDeviceNavCustom(
       mesh_folder=MODEL_PATH,
       model_name="0015_H_AO_COA"
   )
   
   visu = SofaPygame(intervention)
   # ... rest of the code same as dual_human_play_0011.py
   ```

## Creating Training Scripts

To train RL agents on custom models:

```python
from eve_bench.dualdevicenav import DualDeviceNavCustom
from util.env import BenchEnv
from util.agent import BenchAgentSynchron
from eve_rl import Runner

MODEL_PATH = r"D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format"

# Create custom intervention
intervention = DualDeviceNavCustom(
    mesh_folder=MODEL_PATH,
    model_name="0011_H_AO_H"
)

# Wrap in environment
env_train = BenchEnv(
    intervention=intervention,
    mode="train",
    visualisation=False
)

# ... rest of training code same as DualDeviceNav_train.py
```

## Parameters

### `DualDeviceNavCustom`

```python
DualDeviceNavCustom(
    mesh_folder: str,              # Path to dualdevicenav_format folder
    model_name: str = "custom",    # Model name for display
    insertion_point: list = None,  # Custom insertion point [x, y, z] or None for auto-detect
    stop_device_at_tree_end: bool = True,
    normalize_action: bool = False
)
```

**Key Features:**
- **Auto-detection**: Automatically detects insertion point and target branches
- **Error checking**: Validates that all required files exist
- **Flexible**: Works with any VMR model processed in DualDeviceNav format

## Troubleshooting

### Error: "Collision mesh not found"
Make sure you've run the processing script first:
```bash
python vmr_processing_tools\create_dualdevicenav_format.py <vmr_model_folder>
```

### Error: "No centerlines found"
Check that the Centrelines folder contains `.mrk.json` files. The processing might have failed during VMTK extraction.

### Simulation crashes or behaves oddly
- Check that the insertion point is valid (inside the vessel)
- Verify that the centerlines have radius data using:
  ```bash
  python vmr_processing_tools\verify_radius_data.py <model_folder>
  ```

## Batch Processing

To process multiple models at once:

```bash
# Check which models are ready
python vmr_processing_tools\check_vmr_files.py D:\vmr\vmr

# Batch process all ready models
python vmr_processing_tools\batch_create_dualdevicenav.py D:\vmr\vmr
```

## Model Differences

Different VMR models have different characteristics:

| Model Prefix | Anatomy Type | Typical Branches |
|--------------|--------------|------------------|
| `_H` | Healthy | 4-5 |
| `_COA` | Coarctation of Aorta | 4-5 |
| `_MFS` | Marfan Syndrome | 4-6 |
| `_SVD` | ? | 5-6 |
| `_AOD` | Aortic Dissection | Variable |

This affects:
- Number of target branches for navigation
- Vessel geometry and difficulty
- Device behavior and physics

## See Also

- `vmr_processing_tools/RADIUS_COMPLETE_GUIDE.md` - Understanding radius data
- `vmr_processing_tools/EXAMPLE_WORKFLOW.md` - Processing workflow
- `vmr_processing_tools/README.md` - Tools overview


