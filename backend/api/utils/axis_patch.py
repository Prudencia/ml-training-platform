"""
Axis Patch Verification Utility

This module provides functions to verify and apply the Axis patch to YOLOv5 repositories.
The Axis patch modifies the model architecture for compatibility with Axis DLPU:
- Changes activation from SiLU to ReLU6 (quantization-friendly)
- Changes first conv kernel from 6x6 to 5x5
"""

import subprocess
from pathlib import Path
from typing import Dict, Optional
import re

# Axis patch URL
AXIS_PATCH_URL = "https://github.com/AxisCommunications/acap-computer-vision-sdk-examples/raw/main/vdo-larod-inference/models/axis-yolov5-patch.diff"

# Expected values when patch is applied
EXPECTED_ACTIVATION = "nn.ReLU6()"
EXPECTED_FIRST_CONV = "[64, 5, 2, 1]"  # 5x5 kernel with padding 1

# Values when patch is NOT applied (original YOLOv5)
ORIGINAL_ACTIVATION = "nn.SiLU()"
ORIGINAL_FIRST_CONV = "[64, 6, 2, 2]"  # 6x6 kernel with padding 2


def verify_axis_patch(venv_path: str) -> Dict:
    """
    Verify if the Axis patch is applied to the YOLOv5 repository.

    Args:
        venv_path: Path to the virtual environment containing the YOLOv5 repo

    Returns:
        Dict with verification results:
        - is_applied: bool - True if patch is fully applied
        - activation: str - Current activation function
        - first_conv: str - Current first conv layer config
        - details: dict - Per-file verification status
        - message: str - Human-readable status message
    """
    repo_path = Path(venv_path) / "repo"
    models_path = repo_path / "models"

    result = {
        "is_applied": False,
        "activation": None,
        "first_conv": None,
        "details": {},
        "message": ""
    }

    if not models_path.exists():
        result["message"] = f"Models directory not found: {models_path}"
        return result

    # Check common model YAML files
    yaml_files = ["yolov5s.yaml", "yolov5m.yaml", "yolov5l.yaml", "yolov5x.yaml", "yolov5n.yaml"]

    activation_found = None
    first_conv_found = None
    files_checked = 0

    for yaml_file in yaml_files:
        yaml_path = models_path / yaml_file
        if yaml_path.exists():
            files_checked += 1
            with open(yaml_path, 'r') as f:
                content = f.read()

            file_status = {
                "exists": True,
                "has_relu6": False,
                "has_5x5_conv": False
            }

            # Check for activation
            if "activation:" in content:
                activation_match = re.search(r'activation:\s*(.+)', content)
                if activation_match:
                    activation = activation_match.group(1).strip()
                    if activation_found is None:
                        activation_found = activation
                    file_status["has_relu6"] = "ReLU6" in activation

            # Check for first conv layer (5x5 kernel)
            # Pattern: [-1, 1, Conv, [64, 5, 2, 1]]  (patched)
            # vs:      [-1, 1, Conv, [64, 6, 2, 2]]  (original)
            if "Conv, [64, 5, 2, 1]" in content:
                file_status["has_5x5_conv"] = True
                if first_conv_found is None:
                    first_conv_found = EXPECTED_FIRST_CONV
            elif "Conv, [64, 6, 2, 2]" in content:
                if first_conv_found is None:
                    first_conv_found = ORIGINAL_FIRST_CONV

            result["details"][yaml_file] = file_status

    if files_checked == 0:
        result["message"] = "No YAML model files found in repo"
        return result

    result["activation"] = activation_found or "not specified"
    result["first_conv"] = first_conv_found or "not found"

    # Determine overall patch status
    has_relu6 = activation_found and "ReLU6" in activation_found
    has_5x5 = first_conv_found == EXPECTED_FIRST_CONV

    if has_relu6 and has_5x5:
        result["is_applied"] = True
        result["message"] = "Axis patch is APPLIED (ReLU6 activation, 5x5 first conv)"
    elif has_relu6 or has_5x5:
        result["is_applied"] = False
        result["message"] = f"Axis patch is PARTIALLY applied (ReLU6: {has_relu6}, 5x5 conv: {has_5x5})"
    else:
        result["is_applied"] = False
        result["message"] = "Axis patch is NOT applied (using original SiLU/6x6 architecture)"

    return result


def apply_axis_patch(venv_path: str) -> Dict:
    """
    Apply the Axis patch to the YOLOv5 repository.

    Args:
        venv_path: Path to the virtual environment containing the YOLOv5 repo

    Returns:
        Dict with:
        - success: bool
        - message: str
        - verification: dict - Post-apply verification result
    """
    repo_path = Path(venv_path) / "repo"

    if not repo_path.exists():
        return {
            "success": False,
            "message": f"Repository not found: {repo_path}",
            "verification": None
        }

    # First check if patch is already applied
    current_status = verify_axis_patch(venv_path)
    if current_status["is_applied"]:
        return {
            "success": True,
            "message": "Axis patch is already applied",
            "verification": current_status
        }

    try:
        # Download and apply patch
        cmd = f"curl -sL {AXIS_PATCH_URL} | git apply"
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            # Try with --3way for conflicts
            cmd_3way = f"curl -sL {AXIS_PATCH_URL} | git apply --3way"
            result = subprocess.run(
                cmd_3way,
                shell=True,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "message": f"Failed to apply patch: {result.stderr}",
                    "verification": None
                }

        # Verify after applying
        verification = verify_axis_patch(venv_path)

        return {
            "success": verification["is_applied"],
            "message": "Axis patch applied successfully" if verification["is_applied"] else "Patch applied but verification failed",
            "verification": verification
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "message": "Patch application timed out",
            "verification": None
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error applying patch: {str(e)}",
            "verification": None
        }


def get_patch_status_summary(venv_path: str) -> str:
    """
    Get a one-line summary of patch status for logging.

    Args:
        venv_path: Path to the virtual environment

    Returns:
        Summary string suitable for log output
    """
    status = verify_axis_patch(venv_path)

    if status["is_applied"]:
        return f"[AXIS PATCH] VERIFIED - {status['activation']}, first conv: {status['first_conv']}"
    else:
        return f"[AXIS PATCH] NOT APPLIED - {status['activation']}, first conv: {status['first_conv']}"


def check_model_checkpoint_architecture(checkpoint_path: str) -> Optional[Dict]:
    """
    Check the architecture stored in a YOLOv5 checkpoint file.

    This loads the checkpoint's yaml config to verify if the model
    was trained with the Axis patch architecture.

    Args:
        checkpoint_path: Path to .pt checkpoint file

    Returns:
        Dict with architecture info, or None if unable to check
    """
    import sys
    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        return None

    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'model' not in checkpoint:
            return {"error": "No model in checkpoint"}

        model = checkpoint['model']

        # Check for yaml attribute
        yaml_config = None
        if hasattr(model, 'yaml'):
            yaml_config = model.yaml
        elif hasattr(model, 'yaml_file'):
            yaml_config = model.yaml_file

        if yaml_config is None:
            return {"error": "No yaml config in model"}

        # Parse yaml config
        if isinstance(yaml_config, dict):
            activation = yaml_config.get('activation', 'not specified')
            backbone = yaml_config.get('backbone', [])

            # Check first conv layer
            first_conv = "unknown"
            if backbone and len(backbone) > 0:
                first_layer = backbone[0]
                if len(first_layer) >= 4:
                    first_conv = str(first_layer[3])

            has_relu6 = "ReLU6" in str(activation)
            has_5x5 = "5, 2, 1" in first_conv

            return {
                "activation": str(activation),
                "first_conv": first_conv,
                "is_axis_compatible": has_relu6 and has_5x5,
                "message": "Axis-compatible" if (has_relu6 and has_5x5) else "NOT Axis-compatible (retrain required)"
            }

        return {"error": "Unexpected yaml format"}

    except Exception as e:
        return {"error": f"Failed to load checkpoint: {str(e)}"}
