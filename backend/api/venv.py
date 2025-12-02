from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from pathlib import Path
from pydantic import BaseModel
import subprocess
import sys
import os
import threading
from datetime import datetime

from database import get_db, VirtualEnvironment, SessionLocal
from api.utils.axis_patch import verify_axis_patch, apply_axis_patch

router = APIRouter()

VENV_PATH = Path("storage/venvs")
VENV_PATH.mkdir(parents=True, exist_ok=True)
LOG_PATH = Path("storage/logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)

# Track setup status in memory
setup_status: Dict[str, dict] = {}

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
        "apply_axis_patch": True,
        "python_version": "3.9",  # TensorFlow 2.11 requires Python 3.9
        "custom_requirements": "presets/axis_yolov5_requirements.txt"
    },
    "DetectX": {
        "name": "DetectX",
        "description": "DetectX ACAP builder for Axis cameras",
        "github_repo": "https://github.com/pandosme/DetectX.git",
        "apply_axis_patch": False,
        "custom_requirements": "presets/detectx_requirements.txt"
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


def install_python_for_venv(version: str, venv_path: Path):
    """Install specific Python version into venv using pyenv"""
    import glob

    full_version = f"{version}.19" if version == "3.9" else f"{version}.0"

    # Find pyenv
    pyenv_paths = ["/root/.pyenv/bin/pyenv", "/home/*/.pyenv/bin/pyenv", "/usr/local/bin/pyenv"]
    pyenv_bin = None
    for pattern in pyenv_paths:
        matches = glob.glob(pattern)
        if matches:
            pyenv_bin = matches[0]
            break

    if not pyenv_bin:
        raise Exception("pyenv not found. Please install pyenv first.")

    # Get pyenv root
    pyenv_root = subprocess.run(
        [pyenv_bin, "root"],
        capture_output=True, text=True
    ).stdout.strip()

    # Check if version already installed in pyenv
    installed_python = f"{pyenv_root}/versions/{full_version}/bin/python"
    if not os.path.isfile(installed_python):
        # Install Python via pyenv
        subprocess.run(
            [pyenv_bin, "install", "-s", full_version],
            check=True,
            capture_output=True,
            text=True,
            timeout=600
        )

    if not os.path.isfile(installed_python):
        raise Exception(f"Failed to install Python {full_version}")

    return installed_python


def run_setup_in_background(preset_name: str, config: dict, log_file: Path):
    """Run venv setup in background thread with logging"""
    import shutil

    def log(msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        with open(log_file, "a") as f:
            f.write(line)
        setup_status[preset_name]["last_message"] = msg

    try:
        setup_status[preset_name]["status"] = "running"
        setup_status[preset_name]["step"] = "initializing"

        venv_path = VENV_PATH / config["name"]
        venv_path.mkdir(parents=True, exist_ok=True)

        # Get Python executable
        log(f"Starting setup for {preset_name}")
        required_python = config.get("python_version")
        if required_python:
            setup_status[preset_name]["step"] = "installing_python"
            log(f"Installing Python {required_python} via pyenv (this may take several minutes)...")
            try:
                python_exec = install_python_for_venv(required_python, venv_path)
                log(f"Python installed: {python_exec}")
            except Exception as e:
                log(f"ERROR: Failed to install Python {required_python}: {str(e)}")
                if venv_path.exists():
                    shutil.rmtree(venv_path)
                setup_status[preset_name]["status"] = "failed"
                setup_status[preset_name]["error"] = str(e)
                return
        else:
            python_exec = sys.executable
            log(f"Using system Python: {python_exec}")

        # Create venv
        setup_status[preset_name]["step"] = "creating_venv"
        log("Creating virtual environment...")
        try:
            result = subprocess.run(
                [python_exec, "-m", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                check=True
            )
            log("Virtual environment created")
        except subprocess.CalledProcessError as e:
            log(f"ERROR: Failed to create venv: {e.stderr}")
            if venv_path.exists():
                shutil.rmtree(venv_path)
            setup_status[preset_name]["status"] = "failed"
            setup_status[preset_name]["error"] = e.stderr
            return

        python_bin = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"
        result = subprocess.run([str(python_bin), "--version"], capture_output=True, text=True)
        python_version = result.stdout.strip() or result.stderr.strip()
        log(f"Python version: {python_version}")

        # Upgrade pip
        log("Upgrading pip...")
        subprocess.run(
            [str(python_bin), "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True, text=True
        )

        # Clone GitHub repo
        if config.get("github_repo"):
            setup_status[preset_name]["step"] = "cloning_repo"
            repo_path = venv_path / "repo"
            log(f"Cloning repository: {config['github_repo']}")
            try:
                subprocess.run(
                    ["git", "clone", config["github_repo"], str(repo_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                log("Repository cloned successfully")
            except subprocess.CalledProcessError as e:
                log(f"ERROR: Failed to clone repo: {e.stderr}")
                shutil.rmtree(venv_path)
                setup_status[preset_name]["status"] = "failed"
                setup_status[preset_name]["error"] = e.stderr
                return

        # Install requirements
        custom_req = config.get("custom_requirements")
        if custom_req:
            custom_req_path = Path(custom_req)
            if custom_req_path.exists():
                setup_status[preset_name]["step"] = "installing_requirements"
                log(f"Installing requirements from {custom_req} (this may take 10-30 minutes)...")
                log("Installing packages: TensorFlow, PyTorch, OpenCV, etc...")

                # Run pip install with output streaming to log
                try:
                    process = subprocess.Popen(
                        [str(python_bin), "-m", "pip", "install", "-r", str(custom_req_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )

                    # Stream output to log file
                    for line in process.stdout:
                        line = line.strip()
                        if line:
                            # Log significant lines
                            if any(x in line.lower() for x in ['installing', 'collecting', 'downloading', 'error', 'successfully']):
                                log(line)

                    process.wait(timeout=1800)  # 30 min timeout

                    if process.returncode != 0:
                        log(f"ERROR: pip install failed with return code {process.returncode}")
                        shutil.rmtree(venv_path)
                        setup_status[preset_name]["status"] = "failed"
                        setup_status[preset_name]["error"] = f"pip install failed with code {process.returncode}"
                        return

                    log("Requirements installed successfully")

                except subprocess.TimeoutExpired:
                    process.kill()
                    log("ERROR: Installation timed out after 30 minutes")
                    shutil.rmtree(venv_path)
                    setup_status[preset_name]["status"] = "failed"
                    setup_status[preset_name]["error"] = "Installation timed out (>30 min)"
                    return

        # Create database entry
        setup_status[preset_name]["step"] = "finalizing"
        log("Creating database entry...")
        db = SessionLocal()
        try:
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
            setup_status[preset_name]["venv_id"] = new_venv.id
        finally:
            db.close()

        # Apply Axis patch if needed
        if config.get("apply_axis_patch"):
            log("Applying Axis YOLOv5 patch...")
            patch_result = apply_axis_patch(str(venv_path))
            log(f"Patch result: {patch_result}")

        log(f"Setup completed successfully!")
        setup_status[preset_name]["status"] = "completed"
        setup_status[preset_name]["step"] = "done"

    except Exception as e:
        log(f"ERROR: Unexpected error: {str(e)}")
        setup_status[preset_name]["status"] = "failed"
        setup_status[preset_name]["error"] = str(e)


@router.post("/presets/setup/{preset_name}")
async def setup_preset_venv(preset_name: str, db: Session = Depends(get_db)):
    """Start preset virtual environment setup in background"""
    if preset_name not in PRESET_VENVS:
        raise HTTPException(status_code=404, detail=f"Unknown preset: {preset_name}. Available: {list(PRESET_VENVS.keys())}")

    config = PRESET_VENVS[preset_name]

    # Check if already exists in DB
    existing = db.query(VirtualEnvironment).filter(VirtualEnvironment.name == config["name"]).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Virtual environment '{config['name']}' already exists")

    # Check if setup is already running
    if preset_name in setup_status and setup_status[preset_name].get("status") == "running":
        raise HTTPException(status_code=400, detail=f"Setup for '{preset_name}' is already in progress")

    # Check if directory exists but DB entry doesn't (cleanup leftover)
    venv_path = VENV_PATH / config["name"]
    if venv_path.exists():
        import shutil
        shutil.rmtree(venv_path)

    # Create log file
    log_file = LOG_PATH / f"venv_setup_{preset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Initialize status
    setup_status[preset_name] = {
        "status": "starting",
        "step": "initializing",
        "log_file": str(log_file),
        "started_at": datetime.now().isoformat(),
        "last_message": "Starting setup...",
        "error": None,
        "venv_id": None
    }

    # Start background thread
    thread = threading.Thread(
        target=run_setup_in_background,
        args=(preset_name, config, log_file),
        daemon=True
    )
    thread.start()

    return {
        "message": f"Setup started for '{preset_name}'",
        "status": "starting",
        "log_file": str(log_file)
    }


@router.get("/presets/setup/{preset_name}/status")
async def get_setup_status(preset_name: str):
    """Get the current status of a preset setup"""
    if preset_name not in setup_status:
        return {"status": "not_started"}

    status = setup_status[preset_name].copy()

    # Read last few lines of log if available
    log_file = status.get("log_file")
    if log_file and Path(log_file).exists():
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                status["recent_logs"] = [l.strip() for l in lines[-20:]]
        except:
            pass

    return status


@router.get("/presets/setup/{preset_name}/log")
async def get_setup_log(preset_name: str):
    """Get the full setup log for a preset"""
    if preset_name not in setup_status:
        raise HTTPException(status_code=404, detail="No setup found for this preset")

    log_file = setup_status[preset_name].get("log_file")
    if not log_file or not Path(log_file).exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    with open(log_file, "r") as f:
        return {"log": f.read()}


@router.delete("/presets/setup/{preset_name}")
async def cancel_setup(preset_name: str):
    """Cancel a running setup and cleanup"""
    import shutil

    if preset_name in setup_status:
        setup_status[preset_name]["status"] = "cancelled"

    # Remove partial venv directory
    if preset_name in PRESET_VENVS:
        venv_path = VENV_PATH / PRESET_VENVS[preset_name]["name"]
        if venv_path.exists():
            shutil.rmtree(venv_path)

    # Clear status
    if preset_name in setup_status:
        del setup_status[preset_name]

    return {"message": f"Setup for '{preset_name}' cancelled and cleaned up"}
