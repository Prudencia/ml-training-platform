from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel
import subprocess
import sys
import os

from database import get_db, VirtualEnvironment
from api.utils.axis_patch import verify_axis_patch, apply_axis_patch

router = APIRouter()

VENV_PATH = Path("storage/venvs")
VENV_PATH.mkdir(parents=True, exist_ok=True)

class VenvCreate(BaseModel):
    name: str
    description: Optional[str] = None
    github_repo: Optional[str] = None
    custom_setup_commands: Optional[List[str]] = None  # List of commands to run after clone
    python_executable: Optional[str] = None  # Path to specific Python interpreter (e.g., for Python 3.11)

class VenvResponse(BaseModel):
    id: int
    name: str
    path: str
    python_version: str
    created_at: str
    github_repo: Optional[str]
    description: Optional[str]
    is_active: bool

    class Config:
        from_attributes = True

@router.get("/", response_model=List[VenvResponse])
async def list_venvs(db: Session = Depends(get_db)):
    """List all virtual environments"""
    venvs = db.query(VirtualEnvironment).all()
    return [VenvResponse(
        id=v.id,
        name=v.name,
        path=v.path,
        python_version=v.python_version,
        created_at=v.created_at.isoformat(),
        github_repo=v.github_repo,
        description=v.description,
        is_active=v.is_active
    ) for v in venvs]

@router.post("/create")
async def create_venv(venv_data: VenvCreate, db: Session = Depends(get_db)):
    """Create a new virtual environment"""
    # Check if venv name already exists
    existing = db.query(VirtualEnvironment).filter(VirtualEnvironment.name == venv_data.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Virtual environment '{venv_data.name}' already exists")

    venv_path = VENV_PATH / venv_data.name
    venv_path.mkdir(parents=True, exist_ok=True)

    # Determine which Python executable to use
    python_exec = venv_data.python_executable if venv_data.python_executable else sys.executable

    # Create venv using subprocess
    try:
        subprocess.run(
            [python_exec, "-m", "venv", str(venv_path)],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to create venv: {e.stderr}")

    # Get Python version
    python_bin = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"
    result = subprocess.run(
        [str(python_bin), "--version"],
        capture_output=True,
        text=True
    )
    python_version = result.stdout.strip() or result.stderr.strip()

    # If GitHub repo provided, clone it
    if venv_data.github_repo:
        repo_path = venv_path / "repo"
        try:
            subprocess.run(
                ["git", "clone", venv_data.github_repo, str(repo_path)],
                check=True,
                capture_output=True,
                text=True
            )

            # Run custom setup commands if provided
            if venv_data.custom_setup_commands:
                for cmd in venv_data.custom_setup_commands:
                    subprocess.run(
                        cmd,
                        shell=True,
                        check=True,
                        capture_output=True,
                        text=True,
                        cwd=str(repo_path)
                    )

            # Install requirements if they exist
            requirements_file = repo_path / "requirements.txt"
            if requirements_file.exists():
                subprocess.run(
                    [str(python_bin), "-m", "pip", "install", "-r", str(requirements_file)],
                    check=True,
                    capture_output=True,
                    text=True
                )
        except subprocess.CalledProcessError as e:
            # Clean up on failure
            import shutil
            shutil.rmtree(venv_path)
            raise HTTPException(status_code=500, detail=f"Failed to clone/install from GitHub: {e.stderr}")

    # Create database entry
    new_venv = VirtualEnvironment(
        name=venv_data.name,
        path=str(venv_path),
        python_version=python_version,
        github_repo=venv_data.github_repo,
        description=venv_data.description
    )
    db.add(new_venv)
    db.commit()
    db.refresh(new_venv)

    return {"message": "Virtual environment created successfully", "venv_id": new_venv.id}

@router.get("/{venv_id}")
async def get_venv(venv_id: int, db: Session = Depends(get_db)):
    """Get virtual environment details"""
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")
    return venv

@router.get("/{venv_id}/packages")
async def list_packages(venv_id: int, db: Session = Depends(get_db)):
    """List installed packages in virtual environment"""
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    venv_path = Path(venv.path)
    python_bin = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"

    try:
        result = subprocess.run(
            [str(python_bin), "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        import json
        packages = json.loads(result.stdout)
        return {"packages": packages}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to list packages: {e.stderr}")

@router.post("/{venv_id}/install")
async def install_package(venv_id: int, package: str, db: Session = Depends(get_db)):
    """Install a package in virtual environment"""
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    venv_path = Path(venv.path)
    python_bin = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"

    try:
        result = subprocess.run(
            [str(python_bin), "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            check=True
        )
        return {"message": f"Package '{package}' installed successfully", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to install package: {e.stderr}")

@router.get("/{venv_id}/requirements")
async def get_requirements(venv_id: int, db: Session = Depends(get_db)):
    """Generate requirements.txt content for virtual environment"""
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    venv_path = Path(venv.path)
    python_bin = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"

    try:
        result = subprocess.run(
            [str(python_bin), "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        return {"content": result.stdout, "filename": f"{venv.name}_requirements.txt"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate requirements: {e.stderr}")

@router.patch("/{venv_id}/toggle")
async def toggle_venv_active(venv_id: int, db: Session = Depends(get_db)):
    """Toggle the active status of a virtual environment"""
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    # Toggle the is_active status
    venv.is_active = not venv.is_active
    db.commit()
    db.refresh(venv)

    status = "active" if venv.is_active else "inactive"
    return {
        "message": f"Virtual environment '{venv.name}' is now {status}",
        "is_active": venv.is_active
    }

@router.delete("/{venv_id}")
async def delete_venv(venv_id: int, db: Session = Depends(get_db)):
    """Delete a virtual environment"""
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    # Delete files
    import shutil
    venv_path = Path(venv.path)
    if venv_path.exists():
        shutil.rmtree(venv_path)

    # Delete database entry
    db.delete(venv)
    db.commit()

    return {"message": "Virtual environment deleted successfully"}


@router.get("/{venv_id}/axis-patch")
async def get_axis_patch_status(venv_id: int, db: Session = Depends(get_db)):
    """Check if Axis patch is applied to the YOLOv5 repository"""
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    status = verify_axis_patch(venv.path)
    return {
        "venv_id": venv_id,
        "venv_name": venv.name,
        **status
    }


@router.post("/{venv_id}/axis-patch")
async def apply_axis_patch_endpoint(venv_id: int, db: Session = Depends(get_db)):
    """Apply the Axis patch to the YOLOv5 repository"""
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    result = apply_axis_patch(venv.path)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])

    return {
        "venv_id": venv_id,
        "venv_name": venv.name,
        **result
    }


# Preset configurations for special venvs
PRESET_VENVS = {
    "axis_yolov5": {
        "name": "axis_yolov5",
        "description": "YOLOv5 for Axis camera deployment (with ReLU6 patch)",
        "github_repo": "https://github.com/ultralytics/yolov5",
        "apply_axis_patch": True
    },
    "DetectX": {
        "name": "DetectX",
        "description": "DetectX ACAP builder for Axis cameras",
        "github_repo": "https://github.com/pandosme/DetectX.git",
        "apply_axis_patch": False
    }
}


@router.get("/presets/available")
async def get_preset_venvs(db: Session = Depends(get_db)):
    """Get list of available preset venvs and their status"""
    result = []
    for preset_name, config in PRESET_VENVS.items():
        existing = db.query(VirtualEnvironment).filter(VirtualEnvironment.name == preset_name).first()
        result.append({
            "name": preset_name,
            "description": config["description"],
            "exists": existing is not None,
            "venv_id": existing.id if existing else None
        })
    return result


@router.post("/presets/setup/{preset_name}")
async def setup_preset_venv(preset_name: str, db: Session = Depends(get_db)):
    """Create a preset virtual environment with predefined configuration"""
    if preset_name not in PRESET_VENVS:
        raise HTTPException(status_code=404, detail=f"Unknown preset: {preset_name}. Available: {list(PRESET_VENVS.keys())}")

    config = PRESET_VENVS[preset_name]

    # Check if already exists
    existing = db.query(VirtualEnvironment).filter(VirtualEnvironment.name == config["name"]).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Virtual environment '{config['name']}' already exists")

    venv_path = VENV_PATH / config["name"]
    venv_path.mkdir(parents=True, exist_ok=True)

    # Create venv
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to create venv: {e.stderr}")

    # Get Python version
    python_bin = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"
    result = subprocess.run([str(python_bin), "--version"], capture_output=True, text=True)
    python_version = result.stdout.strip() or result.stderr.strip()

    # Clone GitHub repo
    if config.get("github_repo"):
        repo_path = venv_path / "repo"
        try:
            subprocess.run(
                ["git", "clone", config["github_repo"], str(repo_path)],
                check=True,
                capture_output=True,
                text=True
            )

            # Install requirements if they exist
            requirements_file = repo_path / "requirements.txt"
            if requirements_file.exists():
                subprocess.run(
                    [str(python_bin), "-m", "pip", "install", "-r", str(requirements_file)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 min timeout for large installs
                )
        except subprocess.CalledProcessError as e:
            import shutil
            shutil.rmtree(venv_path)
            raise HTTPException(status_code=500, detail=f"Failed to clone/install from GitHub: {e.stderr}")
        except subprocess.TimeoutExpired:
            import shutil
            shutil.rmtree(venv_path)
            raise HTTPException(status_code=500, detail="Installation timed out (>10 min)")

    # Create database entry
    new_venv = VirtualEnvironment(
        name=config["name"],
        path=str(venv_path),
        python_version=python_version,
        github_repo=config.get("github_repo"),
        description=config["description"]
    )
    db.add(new_venv)
    db.commit()
    db.refresh(new_venv)

    # Apply Axis patch if needed
    patch_result = None
    if config.get("apply_axis_patch"):
        patch_result = apply_axis_patch(str(venv_path))

    return {
        "message": f"Preset '{preset_name}' created successfully",
        "venv_id": new_venv.id,
        "patch_applied": patch_result
    }
