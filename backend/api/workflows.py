from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel
import subprocess
import os
import threading
import json
import shutil
from datetime import datetime

from database import get_db, TrainingJob, Export, SessionLocal, VirtualEnvironment

router = APIRouter()

# Storage for inference results
INFERENCE_PATH = Path("storage/inference")
INFERENCE_PATH.mkdir(parents=True, exist_ok=True)

# Track active export processes
active_export_processes = {}

def get_training_final_metrics(job_id: int) -> dict:
    """Get final metrics (mAP, precision, recall) from a training job's results.csv"""
    import csv
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job or not job.model_output_path:
            return None

        model_path = Path(job.model_output_path)
        results_csv = None

        # Find results.csv in subdirectories (YOLOv5 creates experiment folders)
        for csv_file in model_path.glob("*/results.csv"):
            if not results_csv or csv_file.stat().st_mtime > results_csv.stat().st_mtime:
                results_csv = csv_file

        if not results_csv or not results_csv.exists():
            return None

        # Read the last row of results.csv for final metrics
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return None

            last_row = rows[-1]
            # Clean column names (YOLOv5 adds spaces)
            cleaned_row = {k.strip(): v for k, v in last_row.items()}

            return {
                'epochs': len(rows),
                'precision': round(float(cleaned_row.get('metrics/precision', 0)), 3),
                'recall': round(float(cleaned_row.get('metrics/recall', 0)), 3),
                'mAP50': round(float(cleaned_row.get('metrics/mAP_0.5', 0)), 3),
                'mAP50_95': round(float(cleaned_row.get('metrics/mAP_0.5:0.95', 0)), 3)
            }
    except Exception as e:
        print(f"Error getting metrics for job {job_id}: {e}")
        return None
    finally:
        db.close()

class ExportRequest(BaseModel):
    job_id: int
    img_size: int
    format: str = "tflite"
    int8: bool = True
    per_tensor: bool = True
    run_folder: Optional[str] = None  # Optional: which training run folder to export from
    venv_id: Optional[int] = None  # Optional: override venv for export (different GPU configs)

class AxisYOLOv5Setup(BaseModel):
    """Predefined setup for Axis YOLOv5"""
    pass

@router.get("/axis-yolov5/presets")
async def get_axis_yolov5_presets():
    """Get all Axis YOLOv5 preset combinations"""
    models = ["nano", "small", "medium"]
    sizes = [480, 640, 960, 1440]

    # Batch sizes optimized for 8GB GPU (Quadro RTX 4000)
    batch_sizes = {
        "nano": {480: 64, 640: 48, 960: 24, 1440: 8},
        "small": {480: 48, 640: 32, 960: 16, 1440: 6},
        "medium": {480: 32, 640: 16, 960: 8, 1440: 4}
    }

    presets = []
    for model in models:
        for size in sizes:
            model_code = {"nano": "n", "small": "s", "medium": "m"}[model]
            presets.append({
                "name": f"YOLOv5 {model} {size}x{size}",
                "model": model,
                "model_code": model_code,
                "img_size": size,
                "config": {
                    # IMPORTANT: Train from scratch with empty weights to use Axis-patched architecture
                    # Using pre-trained weights (yolov5m.pt) loads the WRONG architecture (6x6 kernel, SiLU)
                    "weights": "",  # Empty = train from scratch
                    "cfg": f"yolov5{model_code}.yaml",  # Use patched YAML config
                    "img": size,
                    "batch": batch_sizes[model][size],
                    "epochs": 300
                }
            })

    return {"presets": presets}

@router.post("/axis-yolov5/setup-venv")
async def setup_axis_yolov5_venv():
    """Return the setup configuration for Axis YOLOv5 virtual environment"""
    import os

    # Check for Python 3.9 via pyenv (absolute path since it's in /root)
    python39_path = "/root/.pyenv/versions/3.9.19/bin/python"

    # Fallback to system Python if pyenv Python 3.9 not found
    if not os.path.exists(python39_path):
        python39_path = None

    return {
        "name": "axis_yolov5",
        "description": "Axis-patched YOLOv5 for ACAP deployment (Python 3.9)",
        "github_repo": "https://github.com/ultralytics/yolov5",
        "python_executable": python39_path,
        "custom_setup_commands": [
            "git checkout 95ebf68f92196975e53ebc7e971d0130432ad107",
            "curl -L https://acap-ml-model-storage.s3.amazonaws.com/yolov5/A9/yolov5-axis-A9.patch | git apply"
        ]
    }

def run_export_in_background(export_id: int, cmd: list, cwd: str, log_path: str, weights_path: str):
    """Run export process in background thread"""
    db = SessionLocal()
    try:
        # Update status to running
        export = db.query(Export).filter(Export.id == export_id).first()
        if export:
            export.status = "running"
            db.commit()

        # Run export
        with open(log_path, 'w') as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                timeout=1800  # 30 minute timeout
            )

        # Check for output file
        export = db.query(Export).filter(Export.id == export_id).first()
        if export:
            if result.returncode == 0:
                # Find the exported tflite file
                weights_dir = Path(weights_path).parent
                tflite_files = list(weights_dir.glob("*.tflite"))
                if tflite_files:
                    # Get the most recent tflite file
                    latest_tflite = max(tflite_files, key=lambda x: x.stat().st_mtime)
                    export.output_path = str(latest_tflite)
                    export.file_size_mb = round(latest_tflite.stat().st_size / (1024 * 1024), 2)
                export.status = "completed"
            else:
                export.status = "failed"
                export.error_message = f"Export failed with return code {result.returncode}"
            export.completed_at = datetime.utcnow()
            db.commit()

    except subprocess.TimeoutExpired:
        export = db.query(Export).filter(Export.id == export_id).first()
        if export:
            export.status = "failed"
            export.error_message = "Export timed out after 30 minutes"
            export.completed_at = datetime.utcnow()
            db.commit()
    except Exception as e:
        export = db.query(Export).filter(Export.id == export_id).first()
        if export:
            export.status = "failed"
            export.error_message = str(e)
            export.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()
        if export_id in active_export_processes:
            del active_export_processes[export_id]

@router.post("/inference/{job_id}")
async def run_inference(
    job_id: int,
    file: UploadFile = File(...),
    conf_thres: float = 0.25,
    db: Session = Depends(get_db)
):
    """Run inference on an uploaded image using a trained model"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Training job is not completed")

    # Get venv for this job
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == job.venv_id).first()
    if not venv:
        raise HTTPException(status_code=404, detail="Virtual environment not found")

    venv_path = Path(venv.path).resolve()
    python_bin = venv_path / "bin" / "python"
    detect_script = venv_path / "repo" / "detect.py"

    if not detect_script.exists():
        raise HTTPException(status_code=400, detail="detect.py not found")

    # Find the best weights
    model_path = Path(job.model_output_path)
    best_weights = None
    weight_candidates = list(model_path.glob("*/weights/best.pt"))
    if weight_candidates:
        best_weights = max(weight_candidates, key=lambda x: x.stat().st_mtime)

    if not best_weights:
        raise HTTPException(status_code=400, detail="No trained weights found")

    # Create unique directory for this inference
    inference_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
    inference_dir = INFERENCE_PATH.resolve() / inference_id
    inference_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded image
    input_image = inference_dir / file.filename
    with open(input_image, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    # Run detection - use absolute paths!
    output_dir = inference_dir / "output"
    cmd = [
        str(python_bin),
        str(detect_script),
        "--weights", str(best_weights),
        "--source", str(input_image.resolve()),  # Absolute path
        "--project", str(output_dir.resolve()),  # Absolute path
        "--name", "detect",
        "--conf-thres", str(conf_thres),
        "--save-txt",
        "--save-conf",
        "--exist-ok"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(venv_path / "repo")
        )

        if result.returncode != 0:
            error_msg = f"Detection failed (code {result.returncode}):\nSTDOUT: {result.stdout[-2000:]}\nSTDERR: {result.stderr[-2000:]}"
            raise HTTPException(status_code=500, detail=error_msg)

        # Parse results
        detections = []
        labels_dir = output_dir / "detect" / "labels"
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            if label_files:
                with open(label_files[0], 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            confidence = float(parts[5]) if len(parts) > 5 else 0.0

                            detections.append({
                                "class_id": class_id,
                                "x_center": x_center,
                                "y_center": y_center,
                                "width": width,
                                "height": height,
                                "confidence": confidence
                            })

        # Get output image path
        output_images = list((output_dir / "detect").glob("*.jpg")) + list((output_dir / "detect").glob("*.png"))
        output_image_url = None
        if output_images:
            output_image_url = f"/api/workflows/inference/result/{inference_id}/{output_images[0].name}"

        return {
            "inference_id": inference_id,
            "detections": detections,
            "num_detections": len(detections),
            "output_image_url": output_image_url,
            "model_name": job.name
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Detection timed out")
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}\n{tb}")

@router.get("/inference/result/{inference_id}/{filename}")
async def get_inference_result_image(inference_id: str, filename: str):
    """Get the output image from inference"""
    image_path = INFERENCE_PATH / inference_id / "output" / "detect" / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Result image not found")

    return FileResponse(str(image_path), media_type="image/jpeg")

@router.post("/compare")
async def compare_models(
    job_ids: str,  # Comma-separated list of job IDs
    file: UploadFile = File(...),
    conf_thres: float = 0.25,
    db: Session = Depends(get_db)
):
    """Compare detection results across multiple models"""
    import time

    # Parse job IDs
    try:
        job_id_list = [int(id.strip()) for id in job_ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job IDs format. Use comma-separated integers.")

    if len(job_id_list) < 2:
        raise HTTPException(status_code=400, detail="At least 2 models required for comparison")

    if len(job_id_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 models can be compared at once")

    # Create comparison directory
    comparison_id = f"compare_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    comparison_dir = INFERENCE_PATH.resolve() / comparison_id
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded image once
    input_image = comparison_dir / file.filename
    with open(input_image, "wb") as f:
        content = await file.read()
        f.write(content)

    results = []

    for job_id in job_id_list:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            results.append({
                "job_id": job_id,
                "error": f"Job {job_id} not found",
                "model_name": "Unknown"
            })
            continue

        if job.status != "completed":
            results.append({
                "job_id": job_id,
                "error": f"Job {job_id} is not completed",
                "model_name": job.name
            })
            continue

        # Find best weights
        model_path = Path(job.model_output_path)
        best_weights = None
        for exp_dir in model_path.glob("*/weights"):
            if (exp_dir / "best.pt").exists():
                best_weights = exp_dir / "best.pt"
                break

        if not best_weights:
            results.append({
                "job_id": job_id,
                "error": "Model weights not found",
                "model_name": job.name
            })
            continue

        # Get venv info
        venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == job.venv_id).first()
        if not venv:
            results.append({
                "job_id": job_id,
                "error": "Virtual environment not found",
                "model_name": job.name
            })
            continue

        venv_path = Path(venv.path).resolve()
        python_bin = venv_path / "bin" / "python"
        detect_script = venv_path / "repo" / "detect.py"

        # Create output directory for this model
        output_dir = comparison_dir / f"model_{job_id}"
        output_dir.mkdir(exist_ok=True)

        # Build detection command
        cmd = [
            str(python_bin), str(detect_script),
            "--weights", str(best_weights),
            "--source", str(input_image.resolve()),
            "--project", str(output_dir.resolve()),
            "--name", "detect",
            "--conf-thres", str(conf_thres),
            "--save-txt", "--save-conf",
            "--exist-ok"
        ]

        # Run inference and measure time
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(venv_path / "repo"),
                capture_output=True,
                text=True,
                timeout=60
            )
            inference_time = time.time() - start_time

            if result.returncode != 0:
                results.append({
                    "job_id": job_id,
                    "error": f"Detection failed: {result.stderr[:500]}",
                    "model_name": job.name,
                    "inference_time_ms": round(inference_time * 1000, 2)
                })
                continue

            # Parse detections from labels
            detections = []
            labels_dir = output_dir / "detect" / "labels"
            if labels_dir.exists():
                label_files = list(labels_dir.glob("*.txt"))
                if label_files:
                    with open(label_files[0], 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 6:
                                detections.append({
                                    "class_id": int(parts[0]),
                                    "x_center": float(parts[1]),
                                    "y_center": float(parts[2]),
                                    "width": float(parts[3]),
                                    "height": float(parts[4]),
                                    "confidence": float(parts[5])
                                })

            # Get output image path
            output_images = list((output_dir / "detect").glob("*.jpg")) + list((output_dir / "detect").glob("*.png"))
            output_image_url = None
            if output_images:
                output_image_url = f"/api/workflows/compare/result/{comparison_id}/model_{job_id}/{output_images[0].name}"

            # Get model info
            model_info = {
                "weights": job.weights or "Unknown",
                "img_size": job.img_size or 640,
                "epochs": job.total_epochs
            }

            results.append({
                "job_id": job_id,
                "model_name": job.name,
                "model_info": model_info,
                "detections": detections,
                "num_detections": len(detections),
                "output_image_url": output_image_url,
                "inference_time_ms": round(inference_time * 1000, 2),
                "avg_confidence": round(sum(d["confidence"] for d in detections) / len(detections), 4) if detections else 0
            })

        except subprocess.TimeoutExpired:
            results.append({
                "job_id": job_id,
                "error": "Detection timed out",
                "model_name": job.name
            })
        except Exception as e:
            results.append({
                "job_id": job_id,
                "error": str(e),
                "model_name": job.name
            })

    return {
        "comparison_id": comparison_id,
        "results": results,
        "input_image": file.filename
    }

@router.get("/compare/result/{comparison_id}/{model_dir}/{filename}")
async def get_comparison_result_image(comparison_id: str, model_dir: str, filename: str):
    """Get the output image from model comparison"""
    image_path = INFERENCE_PATH / comparison_id / model_dir / "detect" / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Result image not found")

    return FileResponse(str(image_path), media_type="image/jpeg")

@router.get("/training/{job_id}/runs")
async def get_training_runs(job_id: int, db: Session = Depends(get_db)):
    """Get all available training runs for a job with their performance metrics"""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    model_output_path = Path(job.model_output_path)
    if not model_output_path.exists():
        return {"runs": [], "count": 0, "job_base_epochs": job.base_epochs or 0}

    runs = []
    # Find all subdirectories with weights/best.pt
    for run_dir in sorted(model_output_path.iterdir()):
        if not run_dir.is_dir():
            continue

        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            continue

        # Try to read metrics from results.csv
        results_csv = run_dir / "results.csv"
        metrics = None

        if results_csv.exists():
            try:
                import csv
                with open(results_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        # Get last row (final epoch)
                        last_row = rows[-1]
                        # Clean column names (they have extra spaces)
                        cleaned = {k.strip(): v.strip() for k, v in last_row.items()}

                        metrics = {
                            "final_epoch": int(float(cleaned.get('epoch', 0))) + 1,
                            "mAP50": round(float(cleaned.get('metrics/mAP_0.5', 0)), 4),
                            "mAP50_95": round(float(cleaned.get('metrics/mAP_0.5:0.95', 0)), 4),
                            "precision": round(float(cleaned.get('metrics/precision', 0)), 4),
                            "recall": round(float(cleaned.get('metrics/recall', 0)), 4),
                            "box_loss": round(float(cleaned.get('train/box_loss', 0)), 5),
                            "obj_loss": round(float(cleaned.get('train/obj_loss', 0)), 5),
                            "cls_loss": round(float(cleaned.get('train/cls_loss', 0)), 5)
                        }
            except Exception as e:
                print(f"Error reading metrics for {run_dir.name}: {e}")

        # Get file modification time as timestamp
        modified_time = best_pt.stat().st_mtime

        # Determine if this is a continued run based on folder name
        is_continued = "_continued" in run_dir.name.lower()

        runs.append({
            "folder_name": run_dir.name,
            "has_best_weights": True,
            "best_weights_path": str(best_pt),
            "modified_timestamp": modified_time,
            "modified_date": datetime.fromtimestamp(modified_time).isoformat(),
            "metrics": metrics,
            "is_continued": is_continued
        })

    # Sort by modification time (most recent first)
    runs.sort(key=lambda x: x["modified_timestamp"], reverse=True)

    return {
        "runs": runs,
        "count": len(runs),
        "job_base_epochs": job.base_epochs or 0  # Current job's base_epochs for calculating effective total
    }

@router.post("/export")
async def export_model(export_req: ExportRequest, db: Session = Depends(get_db)):
    """Export a trained model to TFLite format (async)"""
    job = db.query(TrainingJob).filter(TrainingJob.id == export_req.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # Allow export for completed, paused, or failed jobs as long as best.pt exists
    if job.status not in ["completed", "paused", "failed"]:
        raise HTTPException(status_code=400, detail="Training job must be completed, paused, or failed to export (not running or pending)")

    # Find the best weights
    model_output_path = Path(job.model_output_path)

    # Look for weights - either from specific run folder or find any available
    weights_path = None

    if export_req.run_folder:
        # User specified which run to export
        specific_run = model_output_path / export_req.run_folder / "weights" / "best.pt"
        if specific_run.exists():
            weights_path = specific_run
        else:
            raise HTTPException(status_code=404, detail=f"Weights not found in specified run folder: {export_req.run_folder}")
    else:
        # Find any available weights (backwards compatibility)
        for exp_dir in model_output_path.glob("**/weights"):
            best_pt = exp_dir / "best.pt"
            if best_pt.exists():
                weights_path = best_pt
                break

    if not weights_path:
        raise HTTPException(status_code=404, detail="Model weights (best.pt) not found. Training may not have completed any epochs.")

    # Determine which venv to use (override or training job's venv)
    if export_req.venv_id:
        # Use specified venv for export (by ID)
        venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == export_req.venv_id).first()
        if not venv:
            raise HTTPException(status_code=404, detail="Specified virtual environment not found")
    else:
        # Try to find venv by name first (stable across reinstalls), then fall back to ID
        venv = None
        if job.venv_name:
            venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.name == job.venv_name).first()
        if not venv and job.venv_id:
            venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == job.venv_id).first()
        if not venv:
            raise HTTPException(status_code=404, detail=f"Virtual environment not found (name={job.venv_name}, id={job.venv_id}). Please reinstall the venv.")

    venv_path = Path(venv.path).resolve()
    python_bin = venv_path / "bin" / "python" if os.name != 'nt' else venv_path / "Scripts" / "python.exe"
    export_script = venv_path / "repo" / "export.py"

    if not python_bin.exists():
        raise HTTPException(status_code=404, detail=f"Python binary not found: {python_bin}")
    if not export_script.exists():
        raise HTTPException(status_code=404, detail="export.py not found in virtual environment")

    # Build export command
    cmd = [
        str(python_bin),
        str(export_script),
        "--weights", str(weights_path),
        "--include", export_req.format,
        "--img-size", str(export_req.img_size)
    ]

    if export_req.int8:
        cmd.append("--int8")
    if export_req.per_tensor:
        cmd.append("--per-tensor")

    # Create exports directory
    exports_path = Path("storage/exports")
    exports_path.mkdir(parents=True, exist_ok=True)

    export_log = exports_path / f"export_job_{job.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"

    # Get training metrics snapshot at export time
    metrics = get_training_final_metrics(job.id)

    # Create export record in database with metrics snapshot
    export_record = Export(
        job_id=job.id,
        job_name=job.name,
        status="pending",
        format=export_req.format,
        img_size=export_req.img_size,
        log_path=str(export_log),
        started_at=datetime.utcnow(),
        venv_id=venv.id,  # Store which venv was used for export
        # Save metrics at time of export
        metrics_epochs=metrics.get('epochs') if metrics else None,
        metrics_precision=metrics.get('precision') if metrics else None,
        metrics_recall=metrics.get('recall') if metrics else None,
        metrics_map50=metrics.get('mAP50') if metrics else None,
        metrics_map50_95=metrics.get('mAP50_95') if metrics else None
    )
    db.add(export_record)
    db.commit()
    db.refresh(export_record)

    # Start export in background thread
    thread = threading.Thread(
        target=run_export_in_background,
        args=(export_record.id, cmd, str(venv_path / "repo"), str(export_log), str(weights_path))
    )
    thread.daemon = True
    thread.start()

    active_export_processes[export_record.id] = thread

    return {
        "message": "Export started",
        "export_id": export_record.id,
        "job_id": job.id,
        "log_path": str(export_log)
    }

@router.get("/export-logs/{job_id}")
async def get_export_logs(job_id: int):
    """Get export logs for a specific job"""
    exports_path = Path("storage/exports")
    log_file = exports_path / f"export_job_{job_id}.log"

    if not log_file.exists():
        return {"logs": "Export log not found. Waiting for export to start..."}

    try:
        with open(log_file, 'r') as f:
            logs = f.read()
        return {"logs": logs}
    except Exception as e:
        return {"logs": f"Error reading logs: {str(e)}"}

@router.get("/exports")
async def list_exports(db: Session = Depends(get_db)):
    """List all exports with training metrics (snapshot from export time)"""
    exports = db.query(Export).order_by(Export.started_at.desc()).all()

    result = []
    for e in exports:
        # Use stored metrics (snapshot from export time)
        # Fall back to live metrics for old exports without stored metrics
        if e.metrics_epochs is not None:
            metrics = {
                'epochs': e.metrics_epochs,
                'precision': e.metrics_precision,
                'recall': e.metrics_recall,
                'mAP50': e.metrics_map50,
                'mAP50_95': e.metrics_map50_95
            }
        else:
            # Fallback for old exports: fetch live metrics (will be same for all from same job)
            metrics = get_training_final_metrics(e.job_id)

        result.append({
            "id": e.id,
            "job_id": e.job_id,
            "job_name": e.job_name,
            "status": e.status,
            "format": e.format,
            "img_size": e.img_size,
            "output_path": e.output_path,
            "file_size_mb": e.file_size_mb,
            "log_path": e.log_path,
            "started_at": e.started_at.isoformat() if e.started_at else None,
            "completed_at": e.completed_at.isoformat() if e.completed_at else None,
            "error_message": e.error_message,
            "metrics": metrics
        })

    return {"exports": result}

@router.get("/exports/{export_id}")
async def get_export(export_id: int, db: Session = Depends(get_db)):
    """Get a specific export"""
    export = db.query(Export).filter(Export.id == export_id).first()
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")

    return {
        "id": export.id,
        "job_id": export.job_id,
        "job_name": export.job_name,
        "status": export.status,
        "format": export.format,
        "img_size": export.img_size,
        "output_path": export.output_path,
        "file_size_mb": export.file_size_mb,
        "log_path": export.log_path,
        "started_at": export.started_at.isoformat() if export.started_at else None,
        "completed_at": export.completed_at.isoformat() if export.completed_at else None,
        "error_message": export.error_message
    }

@router.get("/exports/{export_id}/logs")
async def get_export_logs_by_id(export_id: int, db: Session = Depends(get_db)):
    """Get logs for a specific export"""
    export = db.query(Export).filter(Export.id == export_id).first()
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")

    if not export.log_path or not Path(export.log_path).exists():
        return {"logs": "Log file not found. Export may still be starting..."}

    try:
        with open(export.log_path, 'r') as f:
            logs = f.read()
        return {"logs": logs, "status": export.status}
    except Exception as e:
        return {"logs": f"Error reading logs: {str(e)}", "status": export.status}

@router.get("/exports/{export_id}/download")
async def download_export(export_id: int, db: Session = Depends(get_db)):
    """Download the exported model file"""
    export = db.query(Export).filter(Export.id == export_id).first()
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")

    if export.status != "completed":
        raise HTTPException(status_code=400, detail=f"Export is not completed. Status: {export.status}")

    if not export.output_path or not Path(export.output_path).exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    file_path = Path(export.output_path)
    filename = f"{export.job_name}_{export.img_size}px_{export.format}.tflite"
    # Clean filename
    filename = filename.replace(" ", "_").replace("/", "_")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )

@router.delete("/exports/{export_id}")
async def delete_export(export_id: int, db: Session = Depends(get_db)):
    """Delete an export record"""
    export = db.query(Export).filter(Export.id == export_id).first()
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")

    # Optionally delete the file
    if export.output_path and Path(export.output_path).exists():
        try:
            Path(export.output_path).unlink()
        except Exception:
            pass

    if export.log_path and Path(export.log_path).exists():
        try:
            Path(export.log_path).unlink()
        except Exception:
            pass

    db.delete(export)
    db.commit()

    return {"message": "Export deleted successfully"}

# DetectX ACAP Build Configuration
class DetectXBuildRequest(BaseModel):
    export_id: int
    acap_name: str
    friendly_name: str
    version: str
    vendor: str = "Custom"
    vendor_url: str = "https://example.com"
    platform: str = "A8"  # A8, A9, or TPU
    image_size: int = 640  # 480, 640, 768, 960, 1440
    labels: list[str] = None  # Optional custom labels
    objectness: float = 0.25  # Objectness threshold (0.0-1.0), default 0.25 (DetectX default)
    nms: float = 0.05  # IoU threshold for NMS (0.0-1.0), default 0.05 (DetectX default)
    confidence: float = 0.60  # Final confidence threshold (0.0-1.0), default 0.60 for better accuracy

DETECTX_BUILDS_PATH = Path("storage/detectx_builds")
DETECTX_BUILDS_PATH.mkdir(parents=True, exist_ok=True)

def run_detectx_build_in_background(build_id: int, detectx_path: Path, export_path: Path,
                                     labels_path: Path, config: dict, log_path: str):
    """Run DetectX ACAP build in background thread"""
    # Open log file FIRST before any other operations that might fail
    try:
        log_f = open(log_path, 'w')
        log_f.write(f"DetectX ACAP Build Started (Build ID: {build_id})\n")
        log_f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n\n")
        log_f.flush()
    except Exception as e:
        # If we can't even open the log file, there's nothing we can do
        print(f"FATAL: Could not open log file {log_path}: {e}")
        return

    db = SessionLocal()
    try:
        from database import DetectXBuild

        # Update status to running
        build = db.query(DetectXBuild).filter(DetectXBuild.id == build_id).first()
        if build:
            build.status = "running"
            db.commit()
            log_f.write(f"Build record updated to 'running' status\n\n")
            log_f.flush()

        log_f.write(f"About to start build steps...\n")
        log_f.flush()

        try:
            log_f.write(f"Entered inner try block\n")
            log_f.flush()
            log_f.write(f"Starting DetectX ACAP build...\n")
            log_f.write(f"Export: {export_path}\n")
            log_f.write(f"Platform: {config['platform']}\n")
            log_f.write(f"Image Size: {config['image_size']}\n\n")

            # Step 1: Copy model.tflite
            log_f.write("Step 1: Copying model.tflite...\n")
            log_f.write(f"  Source: {export_path}\n")
            log_f.flush()

            model_dest = detectx_path / "app" / "model" / "model.tflite"
            log_f.write(f"  Destination: {model_dest}\n")
            log_f.flush()

            shutil.copy2(export_path, model_dest)
            log_f.write(f"  ✓ Copied successfully\n\n")
            log_f.flush()

            # Step 2: Copy labels.txt
            log_f.write("Step 2: Copying labels.txt...\n")
            labels_dest = detectx_path / "app" / "model" / "labels.txt"
            if labels_path.exists():
                shutil.copy2(labels_path, labels_dest)
                log_f.write(f"  ✓ Copied to {labels_dest}\n\n")
            else:
                log_f.write("  ⚠ Warning: labels.txt not found, using default\n\n")
            log_f.flush()

            # Step 3: Update manifest.json
            log_f.write("Step 3: Updating manifest.json...\n")
            manifest_path = detectx_path / "app" / "manifest.json"
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # NOTE: Keep appName as "detectx" because Makefile is hardcoded to build that binary
            # Only change user-facing metadata
            manifest["acapPackageConf"]["setup"]["friendlyName"] = config["friendly_name"]
            manifest["acapPackageConf"]["setup"]["vendor"] = config["vendor"]
            manifest["acapPackageConf"]["setup"]["vendorUrl"] = config["vendor_url"]
            manifest["acapPackageConf"]["setup"]["version"] = config["version"]

            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=4)
            log_f.write(f"  ✓ Updated manifest.json\n")
            log_f.write(f"    FriendlyName: {config['friendly_name']}\n")
            log_f.write(f"    Version: {config['version']}\n")
            log_f.write(f"    Vendor: {config['vendor']}\n\n")
            log_f.flush()

            # Step 4: Run prepare.py
            log_f.write("Step 4: Running prepare.py to generate model.json...\n")
            venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.name == "DetectX").first()
            if venv:
                # Use absolute path
                python_bin = Path(venv.path).resolve() / "bin" / "python"
                prepare_script = "prepare.py"  # Relative to cwd

                log_f.write(f"Using Python: {python_bin}\n")
                log_f.write(f"Working directory: {detectx_path}\n")

                if not python_bin.exists():
                    log_f.write(f"ERROR: Python not found at {python_bin}\n")
                    raise Exception(f"Python not found at {python_bin}")

                # Run prepare.py with command-line arguments: platform, image_size, objectness, nms, confidence
                log_f.write(f"  Objectness threshold: {config['objectness']}\n")
                log_f.write(f"  NMS threshold: {config['nms']}\n")
                log_f.write(f"  Confidence threshold: {config['confidence']}\n")
                log_f.flush()

                prepare_result = subprocess.run(
                    [str(python_bin), prepare_script,
                     str(config['platform']),
                     str(config['image_size']),
                     str(config['objectness']),
                     str(config['nms']),
                     str(config['confidence'])],
                    capture_output=True,
                    text=True,
                    cwd=str(detectx_path),
                    timeout=60
                )
                log_f.write(prepare_result.stdout)
                if prepare_result.stderr:
                    log_f.write(f"STDERR: {prepare_result.stderr}\n")
                log_f.write("  ✓ prepare.py completed\n\n")
                log_f.flush()
            else:
                log_f.write("  ⚠ Warning: DetectX venv not found, skipping prepare.py\n\n")
                log_f.flush()

            # Step 5: Build Docker image
            log_f.write("Step 5: Building ACAP with Docker...\n")
            log_f.flush()

            build_result = subprocess.run(
                ["docker", "build", "--progress=plain", "--no-cache",
                 "--build-arg", "CHIP=aarch64", ".", "-t", f"detectx_{build_id}"],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(detectx_path),
                timeout=1800  # 30 minutes
            )

            if build_result.returncode != 0:
                raise Exception(f"Docker build failed with return code {build_result.returncode}")

            # Step 6: Extract .eap file from Docker
            log_f.write("\nStep 6: Extracting .eap file from Docker container...\n")

            # Create container
            create_result = subprocess.run(
                ["docker", "create", f"detectx_{build_id}"],
                capture_output=True,
                text=True,
                timeout=60
            )
            container_id = create_result.stdout.strip()
            log_f.write(f"Container ID: {container_id}\n")

            # Copy from container
            build_output = DETECTX_BUILDS_PATH / f"build_{build_id}"
            build_output.mkdir(exist_ok=True)

            subprocess.run(
                ["docker", "cp", f"{container_id}:/opt/app", str(build_output)],
                timeout=120
            )

            # Find .eap file
            eap_files = list((build_output / "app").glob("*.eap"))
            if eap_files:
                eap_file = eap_files[0]
                # Rename with custom name
                final_eap = build_output / f"{config['acap_name']}_{config['version']}.eap"
                shutil.move(str(eap_file), str(final_eap))

                log_f.write(f"\nBuild complete! EAP file: {final_eap.name}\n")
                log_f.write(f"File size: {round(final_eap.stat().st_size / (1024 * 1024), 2)} MB\n")

                # Update database
                build = db.query(DetectXBuild).filter(DetectXBuild.id == build_id).first()
                if build:
                    build.status = "completed"
                    build.output_path = str(final_eap)
                    build.file_size_mb = round(final_eap.stat().st_size / (1024 * 1024), 2)
                    build.completed_at = datetime.utcnow()
                    db.commit()
            else:
                raise Exception("No .eap file found in build output")

            # Cleanup Docker
            subprocess.run(["docker", "rm", container_id], timeout=60)
            subprocess.run(["docker", "rmi", f"detectx_{build_id}"], timeout=60)

        except Exception as e:
            # Inner exception - build steps failed
            import traceback
            log_f.write(f"\n\nERROR: {str(e)}\n")
            log_f.write(traceback.format_exc())
            log_f.flush()

            build = db.query(DetectXBuild).filter(DetectXBuild.id == build_id).first()
            if build:
                build.status = "failed"
                build.error_message = str(e)
                build.completed_at = datetime.utcnow()
                db.commit()

    except Exception as e:
        # Outer exception - database or initialization failed
        import traceback
        log_f.write(f"\n\nFATAL ERROR: {str(e)}\n")
        log_f.write(traceback.format_exc())
        log_f.flush()
    finally:
        log_f.close()
        db.close()

@router.get("/detectx/config")
async def get_detectx_config():
    """Get DetectX build configuration options"""
    return {
        "platforms": [
            {"value": "A8", "label": "ARTPEC-8 DLPU", "chip": "axis-a8-dlpu-tflite"},
            {"value": "A9", "label": "ARTPEC-9 DLPU", "chip": "a9-dlpu-tflite"},
            {"value": "TPU", "label": "Google Edge TPU", "chip": "google-edge-tpu-tflite"}
        ],
        "image_sizes": [
            {"value": 480, "label": "480x480", "video": "640x480"},
            {"value": 640, "label": "640x640", "video": "800x600"},
            {"value": 960, "label": "960x960", "video": "1280x960"},
            {"value": 1440, "label": "1440x1440", "video": "1920x1440"}
        ],
        "default_vendor": "Custom",
        "default_vendor_url": "https://example.com"
    }

@router.get("/detectx/export/{export_id}/labels")
async def get_export_labels(export_id: int, db: Session = Depends(get_db)):
    """Extract labels from an export's training job dataset"""
    export = db.query(Export).filter(Export.id == export_id).first()
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")

    # Get the training job
    job = db.query(TrainingJob).filter(TrainingJob.id == export.job_id).first()
    if not job:
        return {"labels": [], "source": "none"}

    # Try to find labels from the dataset's data.yaml
    if job.config_path:
        config_path = Path(job.config_path)
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    if 'names' in config_data:
                        # names can be a list or a dict
                        if isinstance(config_data['names'], list):
                            return {"labels": config_data['names'], "source": "dataset_yaml"}
                        elif isinstance(config_data['names'], dict):
                            # Convert dict to list sorted by key
                            labels = [config_data['names'][k] for k in sorted(config_data['names'].keys())]
                            return {"labels": labels, "source": "dataset_yaml"}
            except Exception as e:
                pass

    # Fallback: try to read from the model output path
    model_path = Path(job.model_output_path)
    if model_path.exists():
        # Look for labels in various locations
        for labels_file in [
            model_path / "labels.txt",
            model_path.parent / "labels.txt",
        ]:
            if labels_file.exists():
                try:
                    with open(labels_file, 'r') as f:
                        labels = [line.strip() for line in f if line.strip()]
                    return {"labels": labels, "source": "labels_txt"}
                except Exception:
                    pass

    return {"labels": [], "source": "none"}

@router.post("/detectx/build")
async def build_detectx_acap(build_req: DetectXBuildRequest, db: Session = Depends(get_db)):
    """Build DetectX ACAP package from exported model"""
    from database import DetectXBuild

    # Get export
    export = db.query(Export).filter(Export.id == build_req.export_id).first()
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")

    if export.status != "completed":
        raise HTTPException(status_code=400, detail="Export is not completed")

    if not export.output_path or not Path(export.output_path).exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    # Get DetectX venv
    venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.name == "DetectX").first()
    if not venv:
        raise HTTPException(status_code=404, detail="DetectX virtual environment not found. Please create it first.")

    detectx_repo = Path(venv.path) / "repo"
    if not detectx_repo.exists():
        raise HTTPException(status_code=404, detail="DetectX repository not found in venv")

    # Create build directory
    build_id_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
    build_dir = DETECTX_BUILDS_PATH / f"build_{build_id_str}"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Copy DetectX repo to build directory
    detectx_build_path = build_dir / "detectx"
    shutil.copytree(detectx_repo, detectx_build_path)

    # Get labels from request or extract from job
    labels_to_use = build_req.labels
    if not labels_to_use:
        # Try to extract labels from the training job
        job = db.query(TrainingJob).filter(TrainingJob.id == export.job_id).first()
        if job and job.config_path:
            config_path = Path(job.config_path)
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                        if 'names' in config_data:
                            if isinstance(config_data['names'], list):
                                labels_to_use = config_data['names']
                            elif isinstance(config_data['names'], dict):
                                labels_to_use = [config_data['names'][k] for k in sorted(config_data['names'].keys())]
                except Exception as e:
                    pass

    # Create labels.txt
    labels_path = build_dir / "labels.txt"
    if labels_to_use:
        with open(labels_path, 'w') as f:
            for label in labels_to_use:
                f.write(f"{label}\n")
    else:
        # Create default labels file
        labels_to_use = ["object"]
        with open(labels_path, 'w') as f:
            f.write("object\n")

    # Create build record
    build_record = DetectXBuild(
        export_id=build_req.export_id,
        acap_name=build_req.acap_name,
        friendly_name=build_req.friendly_name,
        version=build_req.version,
        platform=build_req.platform,
        image_size=build_req.image_size,
        vendor=build_req.vendor,
        status="pending",
        log_path=str(build_dir / "build.log"),
        started_at=datetime.utcnow()
    )
    db.add(build_record)
    db.commit()
    db.refresh(build_record)

    # Start build in background
    config = {
        "acap_name": build_req.acap_name,
        "friendly_name": build_req.friendly_name,
        "version": build_req.version,
        "vendor": build_req.vendor,
        "vendor_url": build_req.vendor_url,
        "platform": build_req.platform,
        "image_size": build_req.image_size,
        "objectness": build_req.objectness,
        "nms": build_req.nms,
        "confidence": build_req.confidence
    }

    thread = threading.Thread(
        target=run_detectx_build_in_background,
        args=(build_record.id, detectx_build_path, Path(export.output_path),
              labels_path, config, build_record.log_path),
        name=f"DetectX-Build-{build_record.id}"
    )
    # Don't use daemon=True to prevent silent kills on reload
    # thread.daemon = True
    thread.start()

    return {
        "message": "DetectX ACAP build started",
        "build_id": build_record.id,
        "log_path": build_record.log_path
    }

@router.get("/detectx/builds")
async def list_detectx_builds(db: Session = Depends(get_db)):
    """List all DetectX ACAP builds"""
    from database import DetectXBuild

    builds = db.query(DetectXBuild).order_by(DetectXBuild.started_at.desc()).all()
    return {
        "builds": [
            {
                "id": b.id,
                "export_id": b.export_id,
                "acap_name": b.acap_name,
                "friendly_name": b.friendly_name,
                "version": b.version,
                "platform": b.platform,
                "image_size": b.image_size,
                "status": b.status,
                "output_path": b.output_path,
                "file_size_mb": b.file_size_mb,
                "started_at": b.started_at.isoformat() if b.started_at else None,
                "completed_at": b.completed_at.isoformat() if b.completed_at else None,
                "error_message": b.error_message
            }
            for b in builds
        ]
    }

@router.get("/detectx/builds/{build_id}")
async def get_detectx_build(build_id: int, db: Session = Depends(get_db)):
    """Get a specific DetectX build"""
    from database import DetectXBuild

    build = db.query(DetectXBuild).filter(DetectXBuild.id == build_id).first()
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")

    return {
        "id": build.id,
        "export_id": build.export_id,
        "acap_name": build.acap_name,
        "friendly_name": build.friendly_name,
        "version": build.version,
        "platform": build.platform,
        "image_size": build.image_size,
        "vendor": build.vendor,
        "status": build.status,
        "output_path": build.output_path,
        "file_size_mb": build.file_size_mb,
        "log_path": build.log_path,
        "started_at": build.started_at.isoformat() if build.started_at else None,
        "completed_at": build.completed_at.isoformat() if build.completed_at else None,
        "error_message": build.error_message
    }

@router.get("/detectx/builds/{build_id}/logs")
async def get_detectx_build_logs(build_id: int, db: Session = Depends(get_db)):
    """Get logs for a DetectX build"""
    from database import DetectXBuild

    build = db.query(DetectXBuild).filter(DetectXBuild.id == build_id).first()
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")

    if not build.log_path or not Path(build.log_path).exists():
        return {"logs": "Log file not found. Build may still be starting...", "status": build.status}

    try:
        with open(build.log_path, 'r') as f:
            logs = f.read()
        return {"logs": logs, "status": build.status}
    except Exception as e:
        return {"logs": f"Error reading logs: {str(e)}", "status": build.status}

@router.get("/detectx/builds/{build_id}/download")
async def download_detectx_build(build_id: int, db: Session = Depends(get_db)):
    """Download the DetectX .eap file"""
    from database import DetectXBuild

    build = db.query(DetectXBuild).filter(DetectXBuild.id == build_id).first()
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")

    if build.status != "completed":
        raise HTTPException(status_code=400, detail=f"Build is not completed. Status: {build.status}")

    if not build.output_path or not Path(build.output_path).exists():
        raise HTTPException(status_code=404, detail="Build file not found")

    file_path = Path(build.output_path)
    filename = f"{build.acap_name}_{build.version}.eap"

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )

@router.delete("/detectx/builds/{build_id}")
async def delete_detectx_build(build_id: int, db: Session = Depends(get_db)):
    """Delete a DetectX build"""
    from database import DetectXBuild

    build = db.query(DetectXBuild).filter(DetectXBuild.id == build_id).first()
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")

    # Delete output file
    if build.output_path and Path(build.output_path).exists():
        try:
            Path(build.output_path).unlink()
        except Exception:
            pass

    # Delete log file
    if build.log_path and Path(build.log_path).exists():
        try:
            Path(build.log_path).unlink()
        except Exception:
            pass

    db.delete(build)
    db.commit()

    return {"message": "DetectX build deleted successfully"}

@router.get("/workflows/axis-yolov5")
async def get_axis_yolov5_workflow():
    """Get the complete Axis YOLOv5 workflow steps"""
    return {
        "workflow_name": "Axis YOLOv5 Training & Deployment",
        "description": "Complete workflow for training YOLOv5 models for Axis cameras",
        "steps": [
            {
                "step": 1,
                "title": "Create Axis YOLOv5 Virtual Environment",
                "description": "Set up a virtual environment with Axis-patched YOLOv5",
                "endpoint": "/api/venv/create",
                "required_fields": ["name", "github_repo", "custom_setup_commands"],
                "template": {
                    "name": "axis_yolov5",
                    "description": "Axis-patched YOLOv5 for ACAP deployment",
                    "github_repo": "https://github.com/ultralytics/yolov5",
                    "custom_setup_commands": [
                        "git checkout 95ebf68f92196975e53ebc7e971d0130432ad107",
                        "curl -L https://acap-ml-model-storage.s3.amazonaws.com/yolov5/A9/yolov5-axis-A9.patch | git apply"
                    ]
                }
            },
            {
                "step": 2,
                "title": "Upload Dataset",
                "description": "Upload your dataset in YOLOv5 format (images + labels)",
                "endpoint": "/api/datasets/upload",
                "required_fields": ["file", "name", "format"],
                "notes": "Dataset should include data.yaml with paths and class names"
            },
            {
                "step": 3,
                "title": "Create Dataset YAML Config",
                "description": "Create or verify your data.yaml configuration",
                "endpoint": "/api/yaml/",
                "required_fields": ["name", "content", "config_type"],
                "template": {
                    "config_type": "dataset",
                    "content": "path: ../datasets/my_dataset\ntrain: images/train\nval: images/val\nnc: 2\nnames: ['class1', 'class2']"
                }
            },
            {
                "step": 4,
                "title": "Select Training Preset",
                "description": "Choose model size and image resolution",
                "endpoint": "/api/workflows/axis-yolov5/presets",
                "options": "nano/small/medium @ 480x480, 960x960, 1440x1440"
            },
            {
                "step": 5,
                "title": "Start Training",
                "description": "Begin training with selected preset",
                "endpoint": "/api/training/start",
                "required_fields": ["name", "venv_id", "dataset_id", "config_path", "total_epochs"],
                "example_command": "python train.py --img 640 --batch 50 --epochs 300 --data [DATASET]/data.yaml --weights yolov5n.pt --cfg yolov5n.yaml"
            },
            {
                "step": 6,
                "title": "Monitor Training",
                "description": "Watch training progress and logs in real-time",
                "endpoint": "/api/training/{job_id}",
                "websocket": "/api/training/{job_id}/stream"
            },
            {
                "step": 7,
                "title": "Export Model",
                "description": "Export trained model to TFLite INT8 format for Axis cameras",
                "endpoint": "/api/workflows/export",
                "required_fields": ["job_id", "img_size"],
                "example_command": "python export.py --weights runs/train/exp/weights/best.pt --include tflite --int8 --per-tensor --img-size 640",
                "output": "Exported model ready for ACAP deployment"
            }
        ]
    }
