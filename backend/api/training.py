from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel
import subprocess
import signal
import os
import asyncio
import json
from datetime import datetime

from database import get_db, TrainingJob, VirtualEnvironment, Dataset
from api.utils.axis_patch import verify_axis_patch, get_patch_status_summary

router = APIRouter()

LOGS_PATH = Path("storage/logs").resolve()
LOGS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH = Path("storage/models").resolve()
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Store active training processes
active_processes = {}

class TrainingCreate(BaseModel):
    name: str
    venv_id: int
    dataset_id: int
    config_path: str
    total_epochs: int
    batch_size: Optional[int] = 16
    img_size: Optional[int] = 640
    weights: Optional[str] = "yolov5s.pt"
    cfg: Optional[str] = None  # Model config yaml (e.g., "yolov5m.yaml") - required when training from scratch
    device: Optional[str] = "0"  # GPU device, "0" for first GPU, "cpu" for CPU

class TrainingResponse(BaseModel):
    id: int
    name: str
    venv_id: int
    dataset_id: int
    config_path: str
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    current_epoch: int
    total_epochs: int
    base_epochs: int = 0  # Epochs from previous training (for effective total)
    log_path: str
    model_output_path: Optional[str]
    error_message: Optional[str]

    class Config:
        from_attributes = True

@router.get("/", response_model=List[TrainingResponse])
async def list_training_jobs(db: Session = Depends(get_db)):
    """List all training jobs"""
    import re
    jobs = db.query(TrainingJob).all()

    # Check status and parse logs for running jobs
    for j in jobs:
        if j.status == "running":
            # Check if process is still running
            if j.id in active_processes:
                process = active_processes[j.id]
                if process.poll() is not None:  # Process finished
                    j.status = "completed" if process.returncode == 0 else "failed"
                    j.completed_at = datetime.utcnow()
                    if process.returncode != 0:
                        j.error_message = f"Process exited with code {process.returncode}"
                    j.current_epoch = j.total_epochs
                    db.commit()
                    del active_processes[j.id]
            elif j.pid:
                # Process not in active_processes (e.g., after server restart)
                # Check if PID is still alive
                try:
                    os.kill(j.pid, 0)  # Signal 0 just checks if process exists
                except OSError:
                    # Process is dead, check if it completed successfully
                    model_path = Path(j.model_output_path)
                    weights_exist = False
                    for exp_dir in model_path.glob("*/weights"):
                        if (exp_dir / "best.pt").exists():
                            weights_exist = True
                            break

                    if weights_exist:
                        j.status = "completed"
                        j.current_epoch = j.total_epochs
                    else:
                        j.status = "failed"
                        j.error_message = "Process terminated unexpectedly"
                    j.completed_at = datetime.utcnow()
                    db.commit()

            # Parse logs for current epoch if still running
            if j.status == "running" and j.log_path:
                log_file = Path(j.log_path)
                if log_file.exists():
                    try:
                        # Read last 50 lines to find latest epoch
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            # YOLOv5 format: "     56/99      5.65G ..."
                            for line in reversed(lines[-50:]):
                                match = re.search(r'^\s+(\d+)/(\d+)\s+', line)
                                if match:
                                    current = int(match.group(1))
                                    if j.current_epoch != current:
                                        j.current_epoch = current
                                        db.commit()
                                    break
                    except Exception:
                        pass  # Silently fail

    return [TrainingResponse(
        id=j.id,
        name=j.name,
        venv_id=j.venv_id,
        dataset_id=j.dataset_id,
        config_path=j.config_path,
        status=j.status,
        started_at=j.started_at.isoformat() if j.started_at else None,
        completed_at=j.completed_at.isoformat() if j.completed_at else None,
        current_epoch=j.current_epoch,
        total_epochs=j.total_epochs,
        base_epochs=j.base_epochs or 0,
        log_path=j.log_path,
        model_output_path=j.model_output_path,
        error_message=j.error_message
    ) for j in jobs]

@router.post("/start")
async def start_training(training_data: TrainingCreate, db: Session = Depends(get_db)):
    """Start a new training job"""
    # Validate venv exists
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == training_data.venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    # Validate dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == training_data.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Verify Axis patch status
    patch_status = verify_axis_patch(venv.path)
    is_training_from_scratch = not training_data.weights or training_data.weights == ""
    is_using_pretrained = training_data.weights and training_data.weights.endswith('.pt') and not training_data.weights.startswith('/')

    # BLOCK training from scratch if Axis patch is not applied
    if is_training_from_scratch and not patch_status["is_applied"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot train from scratch without Axis patch. {patch_status['message']}. "
                   f"Apply the patch first via /api/venvs/{venv.id}/axis-patch (POST)"
        )

    # WARN/BLOCK when using pre-trained weights (like yolov5m.pt) - these have wrong architecture!
    if is_using_pretrained and not training_data.cfg:
        raise HTTPException(
            status_code=400,
            detail=f"Using pre-trained weights '{training_data.weights}' without --cfg will load the WRONG architecture (6x6 kernel, SiLU). "
                   f"For Axis DLPU compatibility, you must train from scratch: set weights='' and cfg='yolov5m.yaml'"
        )

    # Create log file
    log_file = LOGS_PATH / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    model_output = (MODELS_PATH / training_data.name).resolve()  # Use absolute path

    # Create training job entry
    job = TrainingJob(
        name=training_data.name,
        venv_id=training_data.venv_id,
        dataset_id=training_data.dataset_id,
        config_path=training_data.config_path,
        status="pending",
        total_epochs=training_data.total_epochs,
        log_path=str(log_file),
        model_output_path=str(model_output),
        # Store training parameters for resume
        batch_size=training_data.batch_size,
        img_size=training_data.img_size,
        weights=training_data.weights,
        device=training_data.device
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start training process
    venv_path = Path(venv.path).resolve()  # Convert to absolute path
    python_bin = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"

    # Assume YOLOv5 is in the venv's repo folder
    yolo_script = venv_path / "repo" / "train.py"

    if not yolo_script.exists():
        job.status = "failed"
        job.error_message = "train.py not found in virtual environment"
        db.commit()
        raise HTTPException(status_code=400, detail="train.py not found in virtual environment")

    # Convert config_path to absolute path (resolves relative to backend directory)
    backend_dir = Path(__file__).parent.parent  # Go up from api/ to backend/
    config_path_abs = (backend_dir / training_data.config_path).resolve()

    if not config_path_abs.exists():
        job.status = "failed"
        job.error_message = f"Config file not found: {config_path_abs}"
        db.commit()
        raise HTTPException(status_code=400, detail=f"Config file not found: {config_path_abs}")

    # Build command with GPU and training parameters
    cmd = [
        str(python_bin),
        str(yolo_script),
        "--data", str(config_path_abs),  # Use absolute path
        "--epochs", str(training_data.total_epochs),
        "--batch-size", str(training_data.batch_size),
        "--imgsz", str(training_data.img_size),
        "--weights", training_data.weights if training_data.weights else "",
        "--device", training_data.device,  # Force GPU usage (0) or CPU
        "--project", str(model_output),
        "--name", training_data.name
    ]

    # Add --cfg when training from scratch or when explicitly specified
    if training_data.cfg:
        cmd.extend(["--cfg", training_data.cfg])

    try:
        # Write patch verification header to log
        with open(log_file, 'w') as log_f:
            log_f.write("=" * 60 + "\n")
            log_f.write(get_patch_status_summary(venv.path) + "\n")
            log_f.write(f"Training from scratch: {is_training_from_scratch}\n")
            log_f.write(f"Weights: {training_data.weights or 'None (scratch)'}\n")
            log_f.write(f"Config: {training_data.cfg or 'from weights'}\n")
            log_f.write("=" * 60 + "\n\n")

        # Start process in background (append to log)
        with open(log_file, 'a') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(venv_path / "repo")
            )

        # Store process
        active_processes[job.id] = process

        # Update job
        job.status = "running"
        job.pid = process.pid
        job.started_at = datetime.utcnow()
        db.commit()

        return {
            "message": "Training started successfully",
            "job_id": job.id,
            "pid": process.pid,
            "axis_patch_verified": patch_status["is_applied"],
            "patch_status": patch_status["message"]
        }

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.post("/{job_id}/stop")
async def stop_training(job_id: int, db: Session = Depends(get_db)):
    """Stop a running training job"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status != "running":
        raise HTTPException(status_code=400, detail="Training job is not running")

    # Kill process
    if job_id in active_processes:
        process = active_processes[job_id]
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        del active_processes[job_id]

    job.status = "paused"
    db.commit()

    return {"message": "Training stopped successfully"}

class ResumeTrainingRequest(BaseModel):
    additional_epochs: int = 50
    learning_rate: Optional[float] = 0.001  # Lower LR for fine-tuning (default YOLOv5 is 0.01)

@router.post("/{job_id}/resume")
async def resume_training(job_id: int, resume_data: ResumeTrainingRequest, db: Session = Depends(get_db)):
    """Resume a paused training job OR continue a completed training with additional epochs"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status not in ["paused", "completed", "failed"]:
        raise HTTPException(status_code=400, detail="Training job must be paused, completed, or failed to resume")

    # Find the last checkpoint - search through subdirectories
    model_path = Path(job.model_output_path)
    last_weights = None

    # YOLOv5 creates subdirectories like "name", "name2", "name3", etc.
    # Find the most recent one by checking modification time
    weight_candidates = list(model_path.glob("*/weights/last.pt"))
    if not weight_candidates:
        # Also check directly in case path structure is different
        direct_weights = model_path / "weights" / "last.pt"
        if direct_weights.exists():
            last_weights = direct_weights
    else:
        # Get the most recent weights file
        last_weights = max(weight_candidates, key=lambda x: x.stat().st_mtime)

    if not last_weights or not last_weights.exists():
        raise HTTPException(status_code=400, detail=f"Cannot find last checkpoint in {model_path}")

    # Get venv info
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == job.venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    # Get dataset info for config path
    dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    venv_path = Path(venv.path).resolve()
    python_bin = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"
    yolo_script = venv_path / "repo" / "train.py"

    if not python_bin.exists():
        raise HTTPException(status_code=400, detail=f"Python binary not found: {python_bin}")

    # Convert config_path to absolute path
    backend_dir = Path(__file__).parent.parent
    config_path_abs = (backend_dir / job.config_path).resolve()

    if not config_path_abs.exists():
        raise HTTPException(status_code=400, detail=f"Config file not found: {config_path_abs}")

    # Create new log file for resumed training
    log_file = LOGS_PATH / f"training_{job.id}_resumed_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"

    # CRITICAL: Differentiate between RESUME (paused) and CONTINUE (completed)
    # - PAUSED: Use --resume to continue to ORIGINAL target (no additional epochs)
    # - COMPLETED/FAILED: Use --weights to start fresh training with previous weights (with additional epochs)

    # Store original status before changing it
    original_status = job.status
    is_paused = (original_status == "paused")

    if is_paused:
        # Resume interrupted training: Use --resume flag to continue to ORIGINAL goal
        # YOLOv5 will continue from current epoch to the original total_epochs
        # No additional epochs for paused jobs - just continue to the original target
        cmd = [
            str(python_bin),
            str(yolo_script),
            "--resume", str(last_weights)  # Resume from checkpoint to original target
        ]
        action_message = f"Training resumed from epoch {job.current_epoch} to original target of {job.total_epochs} epochs"
        target_epochs = job.total_epochs  # Keep original target
    else:
        # Continue completed/failed training: Use --weights to start fresh with previous model
        # This avoids YOLOv5's smart_resume error: "training to X epochs is finished, nothing to resume"

        # Store base_epochs from original training for "effective total" tracking
        base_epochs_from_previous = job.base_epochs + job.total_epochs  # Cumulative from all previous training

        # Copy original results.csv to preserve baseline metrics
        original_results = None
        for exp_dir in model_path.glob("*/results.csv"):
            if exp_dir.stat().st_mtime:  # Find most recent results.csv
                original_results = exp_dir
                break

        # Create custom hyperparameters file with lower learning rate for fine-tuning
        # YOLOv5 doesn't support --lr0 on command line, must use --hyp file
        custom_hyp_path = model_path / "hyp_finetune.yaml"
        base_hyp_path = venv_path / "repo" / "data" / "hyps" / "hyp.scratch-low.yaml"

        # Read base hyperparameters and modify lr0
        import yaml
        if base_hyp_path.exists():
            with open(base_hyp_path, 'r') as f:
                hyp_content = yaml.safe_load(f)
        else:
            # Default hyperparameters if file not found
            hyp_content = {
                'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005,
                'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
                'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0,
                'iou_t': 0.2, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h': 0.015,
                'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1,
                'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0,
                'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0
            }

        # Set custom learning rate for fine-tuning
        hyp_content['lr0'] = resume_data.learning_rate

        with open(custom_hyp_path, 'w') as f:
            yaml.dump(hyp_content, f, default_flow_style=False)

        cmd = [
            str(python_bin),
            str(yolo_script),
            "--data", str(config_path_abs),
            "--epochs", str(resume_data.additional_epochs),  # NEW training with additional epochs
            "--batch-size", str(job.batch_size),
            "--imgsz", str(job.img_size),
            "--weights", str(last_weights),  # Start from previous weights (NOT --resume)
            "--device", job.device,
            "--hyp", str(custom_hyp_path),  # Use custom hyp file with lower LR
            "--project", str(model_path),
            "--name", f"{job.name}_continued"  # Different name to avoid overwriting
        ]
        action_message = f"Training continued with {resume_data.additional_epochs} additional epochs using previous weights at LR={resume_data.learning_rate} (effective total: {base_epochs_from_previous + resume_data.additional_epochs} epochs)"
        target_epochs = resume_data.additional_epochs  # New training session with additional epochs

    try:
        # Start process in background
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(venv_path / "repo")
            )

        # Store process
        active_processes[job.id] = process

        # Update job
        job.status = "running"
        job.pid = process.pid
        if not is_paused:
            # For continued training, reset current_epoch since it's a new training session
            job.current_epoch = 0
            # Set base_epochs to track cumulative training
            job.base_epochs = base_epochs_from_previous
        job.total_epochs = target_epochs  # Use calculated target
        job.log_path = str(log_file)
        job.error_message = None
        db.commit()

        return {
            "message": action_message,
            "job_id": job.id,
            "pid": process.pid,
            "new_total_epochs": job.total_epochs
        }

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to resume training: {str(e)}")

@router.get("/{job_id}")
async def get_training_job(job_id: int, db: Session = Depends(get_db)):
    """Get training job details"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Check if process is still running
    if job.id in active_processes:
        process = active_processes[job.id]
        if process.poll() is not None:  # Process finished
            job.status = "completed" if process.returncode == 0 else "failed"
            job.completed_at = datetime.utcnow()
            if process.returncode != 0:
                job.error_message = f"Process exited with code {process.returncode}"
            db.commit()
            del active_processes[job.id]
    elif job.status == "running" and job.pid:
        # Process not in active_processes (e.g., after server restart)
        # Check if PID is still alive
        try:
            os.kill(job.pid, 0)  # Signal 0 just checks if process exists
        except OSError:
            # Process is dead, check if it completed successfully
            # Look for completion markers in log or output files
            model_path = Path(job.model_output_path)
            weights_exist = False
            for exp_dir in model_path.glob("*/weights"):
                if (exp_dir / "best.pt").exists():
                    weights_exist = True
                    break

            if weights_exist:
                job.status = "completed"
                job.current_epoch = job.total_epochs
            else:
                job.status = "failed"
                job.error_message = "Process terminated unexpectedly"
            job.completed_at = datetime.utcnow()
            db.commit()

    # Parse log file to extract current epoch
    if job.status == "running" and job.log_path:
        log_file = Path(job.log_path)
        if log_file.exists():
            try:
                # Read last 50 lines to find latest epoch
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # YOLOv5 format: "     56/99      5.65G ..."
                    # Look for pattern like "NUMBER/NUMBER" at start of line
                    import re
                    for line in reversed(lines[-50:]):
                        match = re.search(r'^\s+(\d+)/(\d+)\s+', line)
                        if match:
                            current = int(match.group(1))
                            total = int(match.group(2))
                            if job.current_epoch != current:
                                job.current_epoch = current
                                db.commit()
                            break
            except Exception as e:
                # Silently fail - don't break the endpoint
                pass

    return job

@router.get("/{job_id}/logs")
async def get_training_logs(job_id: int, lines: int = 100, db: Session = Depends(get_db)):
    """Get training logs (last N lines)"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    log_file = Path(job.log_path)
    if not log_file.exists():
        return {"logs": ""}

    # Read last N lines
    with open(log_file, 'r') as f:
        all_lines = f.readlines()
        last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

    return {"logs": ''.join(last_lines)}

@router.get("/{job_id}/metrics")
async def get_training_metrics(job_id: int, db: Session = Depends(get_db)):
    """Parse training results to extract metrics for visualization"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    import re
    import csv

    metrics = []
    validation = []
    gpu_memory = []
    epoch_times = []

    # First try to read from results.csv (more reliable)
    model_path = Path(job.model_output_path)
    results_csv = None

    # Find results.csv in subdirectories (YOLOv5 creates experiment folders)
    for csv_file in model_path.glob("*/results.csv"):
        if not results_csv or csv_file.stat().st_mtime > results_csv.stat().st_mtime:
            results_csv = csv_file

    if results_csv and results_csv.exists():
        try:
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Clean column names (they have extra spaces)
                    cleaned_row = {k.strip(): v.strip() for k, v in row.items()}

                    epoch = int(float(cleaned_row.get('epoch', 0)))

                    # Training metrics
                    box_loss = float(cleaned_row.get('train/box_loss', 0))
                    obj_loss = float(cleaned_row.get('train/obj_loss', 0))
                    cls_loss = float(cleaned_row.get('train/cls_loss', 0))

                    metrics.append({
                        'epoch': epoch + 1,  # YOLOv5 uses 0-indexed epochs
                        'box_loss': round(box_loss, 6),
                        'obj_loss': round(obj_loss, 6),
                        'cls_loss': round(cls_loss, 6),
                        'total_loss': round(box_loss + obj_loss + cls_loss, 6)
                    })

                    # Validation metrics
                    precision = float(cleaned_row.get('metrics/precision', 0))
                    recall = float(cleaned_row.get('metrics/recall', 0))
                    map50 = float(cleaned_row.get('metrics/mAP_0.5', 0))
                    map50_95 = float(cleaned_row.get('metrics/mAP_0.5:0.95', 0))

                    validation.append({
                        'epoch': epoch + 1,
                        'precision': round(precision, 5),
                        'recall': round(recall, 5),
                        'mAP50': round(map50, 5),
                        'mAP50_95': round(map50_95, 5)
                    })
        except Exception as e:
            print(f"Error reading results.csv: {e}")

    # Parse log file for GPU memory and time estimates
    log_file = Path(job.log_path)
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            seen_epochs = set()
            for line in lines:
                # Parse GPU memory from training lines
                # Format: "     1/10      5.65G ..."
                train_match = re.search(r'^\s+(\d+)/(\d+)\s+(\d+\.?\d*G)', line)
                if train_match:
                    epoch = int(train_match.group(1))
                    gpu_mem = train_match.group(3)
                    gpu_gb = float(gpu_mem.replace('G', ''))

                    if epoch not in seen_epochs:
                        gpu_memory.append({'epoch': epoch, 'memory_gb': gpu_gb})
                        seen_epochs.add(epoch)

                # Parse epoch completion time
                # Format: "10 epochs completed in 0.568 hours."
                time_match = re.search(r'(\d+) epochs? completed in (\d+\.?\d*) hours?', line)
                if time_match:
                    epochs_done = int(time_match.group(1))
                    hours = float(time_match.group(2))
                    epoch_times.append({'epochs': epochs_done, 'hours': hours})
        except Exception as e:
            print(f"Error parsing log file: {e}")

    # Calculate estimated time remaining
    eta_info = {}
    if epoch_times and job.total_epochs:
        last_time = epoch_times[-1]
        hours_per_epoch = last_time['hours'] / last_time['epochs']
        remaining_epochs = job.total_epochs - last_time['epochs']
        if remaining_epochs > 0:
            eta_hours = remaining_epochs * hours_per_epoch
            eta_info = {
                'hours_per_epoch': round(hours_per_epoch, 4),
                'remaining_epochs': remaining_epochs,
                'eta_hours': round(eta_hours, 2),
                'eta_minutes': round(eta_hours * 60, 1)
            }

    return {
        "metrics": metrics,
        "validation": validation,
        "gpu_memory": gpu_memory,
        "epoch_times": epoch_times,
        "eta": eta_info
    }

@router.get("/system/gpu")
async def get_gpu_status():
    """Get current GPU utilization and memory usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 4:
                return {
                    "gpu_utilization": int(parts[0]),
                    "memory_used_mb": int(parts[1]),
                    "memory_total_mb": int(parts[2]),
                    "temperature_c": int(parts[3]),
                    "memory_percent": round(int(parts[1]) / int(parts[2]) * 100, 1)
                }
    except Exception as e:
        pass

    return {
        "gpu_utilization": 0,
        "memory_used_mb": 0,
        "memory_total_mb": 0,
        "temperature_c": 0,
        "memory_percent": 0,
        "error": "GPU info not available"
    }

@router.delete("/{job_id}")
async def delete_training_job(job_id: int, db: Session = Depends(get_db)):
    """Delete a training job"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Stop if running
    if job.status == "running" and job_id in active_processes:
        process = active_processes[job_id]
        process.terminate()
        del active_processes[job_id]

    # Delete database entry
    db.delete(job)
    db.commit()

    return {"message": "Training job deleted successfully"}

@router.get("/queue/list")
async def get_training_queue(db: Session = Depends(get_db)):
    """Get all jobs in the training queue"""
    # Get queued jobs sorted by priority and position
    queued_jobs = db.query(TrainingJob).filter(
        TrainingJob.status == "queued"
    ).order_by(
        TrainingJob.priority.desc(),
        TrainingJob.queue_position.asc()
    ).all()

    # Get currently running job
    running_job = db.query(TrainingJob).filter(
        TrainingJob.status == "running"
    ).first()

    return {
        "running": running_job,
        "queued": queued_jobs,
        "queue_length": len(queued_jobs)
    }

class QueueJobRequest(BaseModel):
    name: str
    venv_id: int
    dataset_id: int
    config_path: str
    total_epochs: int
    batch_size: int = 16
    img_size: int = 640
    weights: str = "yolov5s.pt"
    device: str = "0"
    priority: int = 0

@router.post("/queue/add")
async def add_to_queue(job_data: QueueJobRequest, db: Session = Depends(get_db)):
    """Add a training job to the queue"""
    # Validate venv and dataset exist
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == job_data.venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    dataset = db.query(Dataset).filter(Dataset.id == job_data.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get next queue position
    max_pos = db.query(TrainingJob).filter(
        TrainingJob.status == "queued"
    ).count()

    # Create log file path
    log_file = LOGS_PATH / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    model_output = (MODELS_PATH / job_data.name).resolve()

    # Create queued job
    job = TrainingJob(
        name=job_data.name,
        venv_id=job_data.venv_id,
        dataset_id=job_data.dataset_id,
        config_path=job_data.config_path,
        status="queued",
        total_epochs=job_data.total_epochs,
        log_path=str(log_file),
        model_output_path=str(model_output),
        batch_size=job_data.batch_size,
        img_size=job_data.img_size,
        weights=job_data.weights,
        device=job_data.device,
        queue_position=max_pos + 1,
        priority=job_data.priority
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    return {
        "message": "Job added to queue",
        "job_id": job.id,
        "queue_position": job.queue_position
    }

@router.post("/queue/start-next")
async def start_next_queued_job(db: Session = Depends(get_db)):
    """Start the next job in the queue if no job is currently running"""
    # Check if any job is currently running
    running_job = db.query(TrainingJob).filter(
        TrainingJob.status == "running"
    ).first()

    if running_job:
        return {"message": "A job is already running", "running_job_id": running_job.id}

    # Get next job from queue
    next_job = db.query(TrainingJob).filter(
        TrainingJob.status == "queued"
    ).order_by(
        TrainingJob.priority.desc(),
        TrainingJob.queue_position.asc()
    ).first()

    if not next_job:
        return {"message": "No jobs in queue"}

    # Start the job
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == next_job.venv_id).first()
    if not venv:
        next_job.status = "failed"
        next_job.error_message = "Virtual environment not found"
        db.commit()
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    venv_path = Path(venv.path).resolve()
    python_bin = venv_path / "bin" / "python"
    yolo_script = venv_path / "repo" / "train.py"

    backend_dir = Path(__file__).parent.parent
    config_path_abs = (backend_dir / next_job.config_path).resolve()

    cmd = [
        str(python_bin),
        str(yolo_script),
        "--data", str(config_path_abs),
        "--epochs", str(next_job.total_epochs),
        "--batch-size", str(next_job.batch_size),
        "--imgsz", str(next_job.img_size),
        "--weights", next_job.weights,
        "--device", next_job.device,
        "--project", str(next_job.model_output_path),
        "--name", next_job.name
    ]

    try:
        with open(next_job.log_path, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(venv_path / "repo")
            )

        active_processes[next_job.id] = process

        next_job.status = "running"
        next_job.pid = process.pid
        next_job.started_at = datetime.utcnow()
        next_job.queue_position = None
        db.commit()

        return {
            "message": "Started next job from queue",
            "job_id": next_job.id,
            "job_name": next_job.name,
            "pid": process.pid
        }

    except Exception as e:
        next_job.status = "failed"
        next_job.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to start job: {str(e)}")

@router.post("/queue/reorder")
async def reorder_queue(job_ids: List[int], db: Session = Depends(get_db)):
    """Reorder jobs in the queue"""
    for position, job_id in enumerate(job_ids, start=1):
        job = db.query(TrainingJob).filter(
            TrainingJob.id == job_id,
            TrainingJob.status == "queued"
        ).first()
        if job:
            job.queue_position = position
    db.commit()
    return {"message": "Queue reordered successfully"}

@router.delete("/queue/{job_id}")
async def remove_from_queue(job_id: int, db: Session = Depends(get_db)):
    """Remove a job from the queue"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "queued":
        raise HTTPException(status_code=400, detail="Job is not in queue")

    db.delete(job)
    db.commit()

    return {"message": "Job removed from queue"}

@router.websocket("/{job_id}/stream")
async def stream_logs(websocket: WebSocket, job_id: int, db: Session = Depends(get_db)):
    """WebSocket endpoint for streaming training logs in real-time"""
    await websocket.accept()

    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        await websocket.close(code=404)
        return

    log_file = Path(job.log_path)

    try:
        # Keep track of last position
        last_position = 0

        while True:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    f.seek(last_position)
                    new_content = f.read()
                    if new_content:
                        await websocket.send_json({"type": "log", "content": new_content})
                        last_position = f.tell()

            # Check if job is still running
            if job.status not in ["running", "pending"]:
                await websocket.send_json({"type": "status", "status": job.status})
                break

            await asyncio.sleep(1)  # Poll every second

    except WebSocketDisconnect:
        pass
