from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
import subprocess
import sys
import json
import os
import shutil

from database import (
    get_db,
    AnnotationProject,
    AnnotationClass,
    AnnotationImage,
    Annotation,
    AutoLabelJob,
    AutoLabelPrediction,
    VirtualEnvironment
)

router = APIRouter()

# Use absolute path to avoid issues with subprocess
STORAGE_PATH = Path(__file__).parent.parent / "storage"

# Track paused jobs (job_id -> True if paused)
_paused_jobs = {}


# ============ Pydantic Models ============

class AutoLabelJobCreate(BaseModel):
    project_id: int
    model_path: str  # Path to weights file
    confidence_threshold: float = 0.25
    batch_size: int = 1000  # Number of images to process per batch (100-5000)
    only_unannotated: bool = True  # If True, only process images without annotations


class PredictionAction(BaseModel):
    action: str  # "approve" or "reject"


class BulkPredictionAction(BaseModel):
    prediction_ids: List[int]
    action: str  # "approve", "reject", "approve_all", "reject_all"


class PredictionClassUpdate(BaseModel):
    class_id: int
    class_name: str


class ClassMappingRequest(BaseModel):
    mappings: dict  # {old_class_name: new_class_id}


# ============ Model Discovery ============

# Directory for uploaded pre-trained models
PRETRAINED_MODELS_PATH = STORAGE_PATH / "pretrained_models"
PRETRAINED_MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Supported model file extensions
ALLOWED_MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".tflite", ".weights"}


@router.get("/available-models")
async def get_available_models(db: Session = Depends(get_db)):
    """Get list of available models for auto-labeling.
    Only shows:
    1. Models from completed exports (the best.pt from those training runs)
    2. Uploaded pre-trained models from storage/pretrained_models/
    """
    from database import Export, TrainingJob

    models = []
    seen_paths = set()  # Avoid duplicates

    # 1. Get models from completed exports (find corresponding best.pt)
    exports = db.query(Export).filter(Export.status == "completed").all()

    for export in exports:
        # Get the training job to find the best.pt weights
        job = db.query(TrainingJob).filter(TrainingJob.id == export.job_id).first()
        if not job or not job.model_output_path:
            continue

        model_output_path = Path(job.model_output_path)

        # Find best.pt in the training output
        for weights_path in model_output_path.glob("**/weights/best.pt"):
            if str(weights_path) in seen_paths:
                continue
            seen_paths.add(str(weights_path))

            # Create a descriptive name from the export
            display_name = f"{export.job_name} ({export.img_size}px)"
            if export.metrics_map50:
                display_name += f" - mAP50: {export.metrics_map50:.1%}"

            models.append({
                "name": display_name,
                "path": str(weights_path),
                "source": "export",
                "export_id": export.id,
                "job_id": export.job_id,
                "img_size": export.img_size,
                "size_mb": round(weights_path.stat().st_size / (1024 * 1024), 2) if weights_path.exists() else 0
            })

    # 2. Get uploaded pre-trained models (all supported formats)
    if PRETRAINED_MODELS_PATH.exists():
        for ext in ALLOWED_MODEL_EXTENSIONS:
            for model_file in PRETRAINED_MODELS_PATH.glob(f"*{ext}"):
                if str(model_file) in seen_paths:
                    continue
                seen_paths.add(str(model_file))

                # Try to read metadata file if exists
                meta_file = model_file.with_suffix(".json")
                metadata = {}
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            metadata = json.load(f)
                    except:
                        pass

                display_name = metadata.get("display_name", model_file.stem)
                model_format = ext.replace(".", "").upper()

                models.append({
                    "name": f"{display_name} ({model_format})",
                    "path": str(model_file),
                    "source": "pretrained",
                    "format": model_format,
                    "description": metadata.get("description", "Uploaded pre-trained model"),
                    "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2)
                })

    return {"models": models}


# ============ Pre-trained Model Management ============


@router.get("/pretrained-models")
async def list_pretrained_models():
    """List all uploaded pre-trained models"""
    models = []

    if PRETRAINED_MODELS_PATH.exists():
        # Search for all supported model formats
        for ext in ALLOWED_MODEL_EXTENSIONS:
            for model_file in PRETRAINED_MODELS_PATH.glob(f"*{ext}"):
                # Try to read metadata file if exists
                meta_file = model_file.with_suffix(".json")
                metadata = {}
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            metadata = json.load(f)
                    except:
                        pass

                # Get format type from extension
                format_type = model_file.suffix.upper().replace(".", "")

                models.append({
                    "filename": model_file.name,
                    "display_name": metadata.get("display_name", model_file.stem),
                    "description": metadata.get("description", ""),
                    "format": format_type,
                    "uploaded_at": metadata.get("uploaded_at"),
                    "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                    "path": str(model_file)
                })

    return {"models": models}


@router.post("/pretrained-models/upload")
async def upload_pretrained_model(
    file: UploadFile = File(...),
    display_name: str = Form(...),
    description: str = Form("")
):
    """Upload a pre-trained model (.pt, .onnx, .tflite, etc.)"""
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_MODEL_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(ALLOWED_MODEL_EXTENSIONS)}"
        )

    # Generate a safe filename
    safe_filename = file.filename.replace(" ", "_")
    model_path = PRETRAINED_MODELS_PATH / safe_filename

    # Check for duplicates
    if model_path.exists():
        # Add timestamp to make unique
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{model_path.stem}_{timestamp}{file_ext}"
        model_path = PRETRAINED_MODELS_PATH / safe_filename

    # Save the model file
    try:
        with open(model_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Save metadata
    meta_path = model_path.with_suffix(".json")
    metadata = {
        "display_name": display_name,
        "description": description,
        "original_filename": file.filename,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "message": "Model uploaded successfully",
        "filename": safe_filename,
        "display_name": display_name,
        "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "path": str(model_path)
    }


@router.delete("/pretrained-models/{filename}")
async def delete_pretrained_model(filename: str):
    """Delete a pre-trained model"""
    model_path = PRETRAINED_MODELS_PATH / filename

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    # Validate it's an allowed model format
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_MODEL_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid model file")

    # Delete the model file
    try:
        model_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

    # Delete metadata file if exists
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        try:
            meta_path.unlink()
        except:
            pass

    return {"message": "Model deleted successfully"}


# ============ Auto-Label Job Endpoints ============

@router.post("/jobs")
async def create_autolabel_job(
    job_data: AutoLabelJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create and start an auto-labeling job"""
    # Verify project exists
    project = db.query(AnnotationProject).filter(AnnotationProject.id == job_data.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Verify model exists
    model_path = Path(job_data.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=400, detail="Model file not found")

    # Count images to process based on only_unannotated setting
    if job_data.only_unannotated:
        image_count = db.query(AnnotationImage).filter(
            AnnotationImage.project_id == job_data.project_id,
            AnnotationImage.is_annotated == False
        ).count()
        if image_count == 0:
            raise HTTPException(status_code=400, detail="No unannotated images to process")
    else:
        image_count = db.query(AnnotationImage).filter(
            AnnotationImage.project_id == job_data.project_id
        ).count()
        if image_count == 0:
            raise HTTPException(status_code=400, detail="No images in project")

    # Validate batch_size (clamp to reasonable range)
    batch_size = max(100, min(5000, job_data.batch_size))

    # Create job record
    job = AutoLabelJob(
        project_id=job_data.project_id,
        model_path=str(model_path),
        model_name=model_path.name,
        confidence_threshold=job_data.confidence_threshold,
        batch_size=batch_size,
        only_unannotated=job_data.only_unannotated,
        status="pending",
        total_images=image_count,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start background task
    background_tasks.add_task(run_autolabel_job, job.id)

    return {
        "id": job.id,
        "job_id": job.id,  # Keep for backwards compatibility
        "status": "pending",
        "total_images": image_count,
        "only_unannotated": job_data.only_unannotated,
        "message": "Auto-labeling job started"
    }


@router.get("/jobs")
async def list_autolabel_jobs(
    project_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """List auto-labeling jobs"""
    query = db.query(AutoLabelJob)
    if project_id:
        query = query.filter(AutoLabelJob.project_id == project_id)

    jobs = query.order_by(AutoLabelJob.created_at.desc()).all()

    return [{
        "id": j.id,
        "project_id": j.project_id,
        "model_name": j.model_name,
        "confidence_threshold": j.confidence_threshold,
        "status": j.status,
        "total_images": j.total_images,
        "processed_images": j.processed_images,
        "predictions_count": j.predictions_count,
        "approved_count": j.approved_count,
        "rejected_count": j.rejected_count,
        "started_at": j.started_at.isoformat() if j.started_at else None,
        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        "error_message": j.error_message,
        "created_at": j.created_at.isoformat() if j.created_at else None,
    } for j in jobs]


@router.get("/jobs/{job_id}")
async def get_autolabel_job(job_id: int, db: Session = Depends(get_db)):
    """Get auto-labeling job details"""
    job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "project_id": job.project_id,
        "model_name": job.model_name,
        "model_path": job.model_path,
        "confidence_threshold": job.confidence_threshold,
        "status": job.status,
        "total_images": job.total_images,
        "processed_images": job.processed_images,
        "predictions_count": job.predictions_count,
        "approved_count": job.approved_count,
        "rejected_count": job.rejected_count,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


@router.delete("/jobs/{job_id}")
async def delete_autolabel_job(job_id: int, db: Session = Depends(get_db)):
    """Delete an auto-labeling job and its predictions"""
    job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete predictions
    db.query(AutoLabelPrediction).filter(AutoLabelPrediction.job_id == job_id).delete()

    # Delete job
    db.delete(job)
    db.commit()

    # Clean up paused state
    if job_id in _paused_jobs:
        del _paused_jobs[job_id]

    return {"message": "Job deleted"}


@router.post("/jobs/{job_id}/pause")
async def pause_autolabel_job(job_id: int, db: Session = Depends(get_db)):
    """Pause a running auto-labeling job"""
    job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "running":
        raise HTTPException(status_code=400, detail="Job is not running")

    _paused_jobs[job_id] = True
    job.status = "paused"
    db.commit()

    return {"message": "Job paused", "status": "paused"}


@router.post("/jobs/{job_id}/resume")
async def resume_autolabel_job(
    job_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Resume a paused auto-labeling job"""
    job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "paused":
        raise HTTPException(status_code=400, detail="Job is not paused")

    # Clear paused state
    if job_id in _paused_jobs:
        del _paused_jobs[job_id]

    job.status = "running"
    db.commit()

    # Resume background task
    background_tasks.add_task(run_autolabel_job, job_id)

    return {"message": "Job resumed", "status": "running"}


# ============ Prediction Endpoints ============

@router.get("/jobs/{job_id}/predictions")
async def get_predictions(
    job_id: int,
    status: Optional[str] = None,
    image_id: Optional[int] = None,
    page: int = 1,
    per_page: int = 50,
    db: Session = Depends(get_db)
):
    """Get predictions for a job"""
    query = db.query(AutoLabelPrediction).filter(AutoLabelPrediction.job_id == job_id)

    if status:
        query = query.filter(AutoLabelPrediction.status == status)
    if image_id:
        query = query.filter(AutoLabelPrediction.image_id == image_id)

    total = query.count()
    predictions = query.order_by(AutoLabelPrediction.image_id, AutoLabelPrediction.id)\
        .offset((page - 1) * per_page).limit(per_page).all()

    return {
        "predictions": [{
            "id": p.id,
            "image_id": p.image_id,
            "class_id": p.class_id,
            "class_name": p.class_name,
            "confidence": p.confidence,
            "x_center": p.x_center,
            "y_center": p.y_center,
            "width": p.width,
            "height": p.height,
            "status": p.status,
        } for p in predictions],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.get("/jobs/{job_id}/images-with-predictions")
async def get_images_with_predictions(
    job_id: int,
    db: Session = Depends(get_db)
):
    """Get list of images that have predictions from this job"""
    job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get unique image IDs with predictions
    image_ids = db.query(AutoLabelPrediction.image_id).filter(
        AutoLabelPrediction.job_id == job_id
    ).distinct().all()
    image_ids = [i[0] for i in image_ids]

    # Get image details with prediction counts
    images = []
    for img_id in image_ids:
        img = db.query(AnnotationImage).filter(AnnotationImage.id == img_id).first()
        if img:
            pending_count = db.query(AutoLabelPrediction).filter(
                AutoLabelPrediction.job_id == job_id,
                AutoLabelPrediction.image_id == img_id,
                AutoLabelPrediction.status == "pending"
            ).count()

            approved_count = db.query(AutoLabelPrediction).filter(
                AutoLabelPrediction.job_id == job_id,
                AutoLabelPrediction.image_id == img_id,
                AutoLabelPrediction.status == "approved"
            ).count()

            images.append({
                "id": img.id,
                "filename": img.filename,
                "original_width": img.original_width,
                "original_height": img.original_height,
                "pending_predictions": pending_count,
                "approved_predictions": approved_count,
            })

    return {"images": images}


@router.put("/predictions/{prediction_id}")
async def update_prediction(
    prediction_id: int,
    action: PredictionAction,
    db: Session = Depends(get_db)
):
    """Approve or reject a single prediction"""
    prediction = db.query(AutoLabelPrediction).filter(AutoLabelPrediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    job = db.query(AutoLabelJob).filter(AutoLabelJob.id == prediction.job_id).first()

    if action.action == "approve":
        # Create annotation from prediction
        image = db.query(AnnotationImage).filter(AnnotationImage.id == prediction.image_id).first()

        new_annotation = Annotation(
            image_id=prediction.image_id,
            class_id=prediction.class_id,
            x_center=prediction.x_center,
            y_center=prediction.y_center,
            width=prediction.width,
            height=prediction.height,
        )
        db.add(new_annotation)

        # Update image status
        if image:
            image.annotation_count += 1
            image.is_annotated = True

        prediction.status = "approved"
        if job:
            job.approved_count += 1

    elif action.action == "reject":
        prediction.status = "rejected"
        if job:
            job.rejected_count += 1
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'approve' or 'reject'")

    db.commit()

    # Update project counts
    if prediction.image:
        _update_project_counts(prediction.image.project_id, db)

    return {"message": f"Prediction {action.action}d"}


@router.post("/predictions/bulk")
async def bulk_update_predictions(
    data: BulkPredictionAction,
    db: Session = Depends(get_db)
):
    """Bulk approve or reject predictions"""
    if data.action not in ["approve", "reject"]:
        raise HTTPException(status_code=400, detail="Invalid action")

    predictions = db.query(AutoLabelPrediction).filter(
        AutoLabelPrediction.id.in_(data.prediction_ids),
        AutoLabelPrediction.status == "pending"
    ).all()

    approved_count = 0
    rejected_count = 0

    for prediction in predictions:
        job = db.query(AutoLabelJob).filter(AutoLabelJob.id == prediction.job_id).first()

        if data.action == "approve":
            # Create annotation
            image = db.query(AnnotationImage).filter(AnnotationImage.id == prediction.image_id).first()

            new_annotation = Annotation(
                image_id=prediction.image_id,
                class_id=prediction.class_id,
                x_center=prediction.x_center,
                y_center=prediction.y_center,
                width=prediction.width,
                height=prediction.height,
            )
            db.add(new_annotation)

            if image:
                image.annotation_count += 1
                image.is_annotated = True

            prediction.status = "approved"
            if job:
                job.approved_count += 1
            approved_count += 1

        elif data.action == "reject":
            prediction.status = "rejected"
            if job:
                job.rejected_count += 1
            rejected_count += 1

    db.commit()

    # Update project counts for affected projects
    project_ids = set()
    for prediction in predictions:
        if prediction.image:
            project_ids.add(prediction.image.project_id)

    for project_id in project_ids:
        _update_project_counts(project_id, db)

    return {
        "approved": approved_count,
        "rejected": rejected_count,
    }


@router.patch("/predictions/{prediction_id}/class")
async def update_prediction_class(
    prediction_id: int,
    data: PredictionClassUpdate,
    db: Session = Depends(get_db)
):
    """Update a prediction's class before approving"""
    prediction = db.query(AutoLabelPrediction).filter(AutoLabelPrediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if prediction.status != "pending":
        raise HTTPException(status_code=400, detail="Can only update class for pending predictions")

    prediction.class_id = data.class_id
    prediction.class_name = data.class_name
    db.commit()

    return {"message": "Prediction class updated", "class_id": data.class_id, "class_name": data.class_name}


@router.post("/jobs/{job_id}/apply-class-mapping")
async def apply_class_mapping(
    job_id: int,
    data: ClassMappingRequest,
    db: Session = Depends(get_db)
):
    """Apply class mappings to all pending predictions in a job.

    mappings format: {old_class_name: {class_id: int, class_name: str}}
    """
    job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get all pending predictions
    predictions = db.query(AutoLabelPrediction).filter(
        AutoLabelPrediction.job_id == job_id,
        AutoLabelPrediction.status == "pending"
    ).all()

    updated_count = 0
    for prediction in predictions:
        if prediction.class_name in data.mappings:
            mapping = data.mappings[prediction.class_name]
            if isinstance(mapping, dict) and 'class_id' in mapping:
                prediction.class_id = mapping['class_id']
                prediction.class_name = mapping.get('class_name', prediction.class_name)
                updated_count += 1

    db.commit()

    return {"message": f"Updated {updated_count} predictions", "updated_count": updated_count}


@router.get("/jobs/{job_id}/unique-classes")
async def get_job_unique_classes(job_id: int, db: Session = Depends(get_db)):
    """Get unique class names from all predictions in a job"""
    job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get unique class names and their counts
    results = db.query(
        AutoLabelPrediction.class_name,
        func.count(AutoLabelPrediction.id).label('count')
    ).filter(
        AutoLabelPrediction.job_id == job_id,
        AutoLabelPrediction.status == "pending"
    ).group_by(AutoLabelPrediction.class_name).all()

    return {
        "classes": [
            {"name": r.class_name, "count": r.count}
            for r in results
        ]
    }


@router.post("/jobs/{job_id}/approve-all")
async def approve_all_predictions(job_id: int, db: Session = Depends(get_db)):
    """Approve all pending predictions for a job"""
    job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    predictions = db.query(AutoLabelPrediction).filter(
        AutoLabelPrediction.job_id == job_id,
        AutoLabelPrediction.status == "pending"
    ).all()

    count = 0
    for prediction in predictions:
        image = db.query(AnnotationImage).filter(AnnotationImage.id == prediction.image_id).first()

        new_annotation = Annotation(
            image_id=prediction.image_id,
            class_id=prediction.class_id,
            x_center=prediction.x_center,
            y_center=prediction.y_center,
            width=prediction.width,
            height=prediction.height,
        )
        db.add(new_annotation)

        if image:
            image.annotation_count += 1
            image.is_annotated = True

        prediction.status = "approved"
        count += 1

    job.approved_count += count
    db.commit()

    # Update project counts
    _update_project_counts(job.project_id, db)

    return {"approved": count}


@router.post("/jobs/{job_id}/reject-all")
async def reject_all_predictions(job_id: int, db: Session = Depends(get_db)):
    """Reject all pending predictions for a job"""
    job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = db.query(AutoLabelPrediction).filter(
        AutoLabelPrediction.job_id == job_id,
        AutoLabelPrediction.status == "pending"
    ).update({"status": "rejected"})

    job.rejected_count += result
    db.commit()

    return {"rejected": result}


# ============ Background Task: Run Inference ============

def run_onnx_inference(model_path: Path, image_path: Path, conf_threshold: float = 0.25):
    """Run ONNX model inference using onnxruntime.

    Returns list of detections: [(class_id, x_center, y_center, width, height, confidence), ...]
    """
    try:
        import onnxruntime as ort
        import numpy as np
        from PIL import Image
    except ImportError:
        raise RuntimeError("onnxruntime or PIL not installed. Install with: pip install onnxruntime pillow")

    # Load model
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

    # Get model input details
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # e.g., [1, 3, 640, 640]

    # Handle dynamic dimensions
    if isinstance(input_shape[2], str) or input_shape[2] is None:
        img_size = 640  # Default
    else:
        img_size = input_shape[2]

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    orig_width, orig_height = img.size

    # Resize with letterbox (maintain aspect ratio)
    scale = min(img_size / orig_width, img_size / orig_height)
    new_w, new_h = int(orig_width * scale), int(orig_height * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Create padded image
    img_padded = Image.new('RGB', (img_size, img_size), (114, 114, 114))
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    img_padded.paste(img_resized, (pad_x, pad_y))

    # Convert to numpy and normalize
    img_np = np.array(img_padded).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    img_np = np.expand_dims(img_np, 0)  # Add batch dimension

    # Run inference
    outputs = session.run(None, {input_name: img_np})

    # Process outputs (YOLOv5/v8 format)
    detections = []
    output = outputs[0]

    # Handle different output formats
    if len(output.shape) == 3:
        # Shape: [1, num_boxes, 85] (YOLOv5) or [1, 84, num_boxes] (YOLOv8)
        if output.shape[1] > output.shape[2]:
            # YOLOv8 format: [1, 84, 8400] -> transpose to [1, 8400, 84]
            output = output.transpose(0, 2, 1)

        preds = output[0]  # Remove batch dimension

        for pred in preds:
            # YOLOv5 format: [x, y, w, h, obj_conf, class1_conf, class2_conf, ...]
            # YOLOv8 format: [x, y, w, h, class1_conf, class2_conf, ...]

            if len(pred) > 5:
                # Check if it has objectness score (YOLOv5) or not (YOLOv8)
                if len(pred) == 85:  # YOLOv5 with 80 classes + 5 (x,y,w,h,obj)
                    obj_conf = pred[4]
                    class_scores = pred[5:]
                    if obj_conf < conf_threshold:
                        continue
                    class_id = int(np.argmax(class_scores))
                    class_conf = class_scores[class_id]
                    confidence = float(obj_conf * class_conf)
                else:  # YOLOv8 format
                    class_scores = pred[4:]
                    class_id = int(np.argmax(class_scores))
                    confidence = float(class_scores[class_id])

                if confidence < conf_threshold:
                    continue

                # Get box coordinates (in model input space)
                x_center_px = pred[0]
                y_center_px = pred[1]
                width_px = pred[2]
                height_px = pred[3]

                # Convert from padded image coordinates to original image coordinates
                x_center_px = (x_center_px - pad_x) / scale
                y_center_px = (y_center_px - pad_y) / scale
                width_px = width_px / scale
                height_px = height_px / scale

                # Normalize to 0-1 range (YOLO format)
                x_center = float(x_center_px / orig_width)
                y_center = float(y_center_px / orig_height)
                width = float(width_px / orig_width)
                height = float(height_px / orig_height)

                # Clamp values
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                if width > 0 and height > 0:
                    detections.append((class_id, x_center, y_center, width, height, confidence))

    # Apply NMS
    if detections:
        detections = apply_nms(detections, iou_threshold=0.45)

    return detections


def apply_nms(detections, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to detections."""
    if not detections:
        return []

    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x[5], reverse=True)

    def iou(box1, box2):
        """Calculate IoU between two boxes in (x_center, y_center, w, h) format."""
        x1_min = box1[1] - box1[3] / 2
        y1_min = box1[2] - box1[4] / 2
        x1_max = box1[1] + box1[3] / 2
        y1_max = box1[2] + box1[4] / 2

        x2_min = box2[1] - box2[3] / 2
        y2_min = box2[2] - box2[4] / 2
        x2_max = box2[1] + box2[3] / 2
        y2_max = box2[2] + box2[4] / 2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = box1[3] * box1[4]
        box2_area = box2[3] * box2[4]
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if iou(best, d) < iou_threshold]

    return kept


def run_autolabel_job(job_id: int):
    """Run inference on unlabeled images using YOLOv5 detect.py or ONNX runtime.

    Uses batch processing for YOLOv5 .pt models to dramatically improve performance
    by loading the model once per batch instead of once per image.
    """
    import time
    from database import SessionLocal

    db = SessionLocal()
    try:
        job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
        if not job:
            return

        # Use configurable batch size from job, with fallback to 1000
        BATCH_SIZE = job.batch_size if job.batch_size else 1000

        job.status = "running"
        if not job.started_at:
            job.started_at = datetime.utcnow()
        db.commit()

        # Get project classes
        project = db.query(AnnotationProject).filter(AnnotationProject.id == job.project_id).first()
        project_classes = {c.class_index: c.name for c in project.classes}

        # Get images that already have predictions from this job (for resume)
        already_processed_ids = set(
            p[0] for p in db.query(AutoLabelPrediction.image_id).filter(
                AutoLabelPrediction.job_id == job_id
            ).distinct().all()
        )

        # Get images based on only_unannotated setting, excluding already processed ones
        # Handle backwards compatibility: if only_unannotated is None, default to True
        only_unannotated = job.only_unannotated if job.only_unannotated is not None else True

        if only_unannotated:
            # Only process unannotated images
            images = db.query(AnnotationImage).filter(
                AnnotationImage.project_id == job.project_id,
                AnnotationImage.is_annotated == False,
                ~AnnotationImage.id.in_(already_processed_ids) if already_processed_ids else True
            ).all()
        else:
            # Process all images (including already annotated ones)
            images = db.query(AnnotationImage).filter(
                AnnotationImage.project_id == job.project_id,
                ~AnnotationImage.id.in_(already_processed_ids) if already_processed_ids else True
            ).all()

        model_path = Path(job.model_path)
        model_ext = model_path.suffix.lower()

        # Determine inference method based on model type
        use_onnx = model_ext == ".onnx"
        yolov5_path = None
        python_path = None

        if not use_onnx:
            # Find YOLOv5 detect.py for .pt files
            venvs_path = STORAGE_PATH / "venvs"
            if venvs_path.exists():
                for venv_dir in venvs_path.iterdir():
                    if venv_dir.is_dir():
                        repo_detect = venv_dir / "repo" / "detect.py"
                        if repo_detect.exists():
                            yolov5_path = (venv_dir / "repo").resolve()
                            python_path = (venv_dir / "bin" / "python").absolute()
                            break
                        direct_detect = venv_dir / "detect.py"
                        if direct_detect.exists():
                            yolov5_path = venv_dir.resolve()
                            python_path = (venv_dir / "bin" / "python").absolute()
                            break

            if not yolov5_path:
                job.status = "failed"
                job.error_message = "YOLOv5 not found in virtual environments. For .pt models, you need a venv with YOLOv5 repo cloned."
                job.completed_at = datetime.utcnow()
                db.commit()
                return

            if not python_path or not python_path.exists():
                python_path = Path(sys.executable)

        predictions_count = job.predictions_count or 0
        base_processed = len(already_processed_ids)

        # Build image lookup dict (filename -> image object)
        image_lookup = {}
        for img in images:
            img_path = STORAGE_PATH / img.file_path
            if img_path.exists():
                image_lookup[img_path.stem] = img

        # Process in batches for .pt models (ONNX still uses per-image)
        if use_onnx:
            # ONNX: process one at a time (already fast enough)
            for idx, image in enumerate(images):
                if _paused_jobs.get(job_id):
                    print(f"Job {job_id} paused, stopping processing")
                    return

                try:
                    image_path = STORAGE_PATH / image.file_path
                    if not image_path.exists():
                        continue

                    detections = run_onnx_inference(model_path, image_path, job.confidence_threshold)

                    for det in detections:
                        class_id, x_center, y_center, width, height, confidence = det
                        class_name = project_classes.get(class_id, f"class_{class_id}")
                        prediction = AutoLabelPrediction(
                            job_id=job_id, image_id=image.id, class_id=class_id,
                            class_name=class_name, confidence=confidence,
                            x_center=x_center, y_center=y_center,
                            width=width, height=height, status="pending",
                        )
                        db.add(prediction)
                        predictions_count += 1

                    job.processed_images = base_processed + idx + 1
                    job.predictions_count = predictions_count
                    db.commit()
                except Exception as e:
                    print(f"ONNX inference error on image {image.id}: {e}")
                    continue
        else:
            # YOLOv5 .pt models: use batch processing for 80x speedup
            image_list = list(image_lookup.keys())
            total_batches = (len(image_list) + BATCH_SIZE - 1) // BATCH_SIZE

            for batch_idx in range(total_batches):
                # Check if job was paused
                if _paused_jobs.get(job_id):
                    print(f"Job {job_id} paused at batch {batch_idx + 1}/{total_batches}")
                    return

                batch_start = batch_idx * BATCH_SIZE
                batch_end = min(batch_start + BATCH_SIZE, len(image_list))
                batch_stems = image_list[batch_start:batch_end]

                # Create temp directory with symlinks for this batch
                batch_dir = Path(f"/tmp/autolabel/job_{job_id}_batch")
                if batch_dir.exists():
                    shutil.rmtree(batch_dir)
                batch_dir.mkdir(parents=True, exist_ok=True)

                # Create symlinks for batch images
                for stem in batch_stems:
                    img = image_lookup[stem]
                    src_path = STORAGE_PATH / img.file_path
                    dst_path = batch_dir / src_path.name
                    try:
                        dst_path.symlink_to(src_path)
                    except FileExistsError:
                        pass

                # Run YOLOv5 on entire batch directory
                output_dir = Path(f"/tmp/autolabel/job_{job_id}_out")
                cmd = [
                    str(python_path),
                    str(yolov5_path / "detect.py"),
                    "--weights", str(model_path),
                    "--source", str(batch_dir),
                    "--conf-thres", str(job.confidence_threshold),
                    "--save-txt",
                    "--save-conf",
                    "--nosave",
                    "--project", str(output_dir.parent),
                    "--name", output_dir.name,
                    "--exist-ok",
                ]

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=600,  # 10 min timeout for batch
                        cwd=str(yolov5_path)
                    )

                    # Parse all label files from this batch
                    labels_dir = output_dir / "labels"
                    if labels_dir.exists():
                        for label_file in labels_dir.glob("*.txt"):
                            stem = label_file.stem
                            if stem not in image_lookup:
                                continue

                            img = image_lookup[stem]
                            with open(label_file) as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])
                                        x_center = float(parts[1])
                                        y_center = float(parts[2])
                                        width = float(parts[3])
                                        height = float(parts[4])
                                        confidence = float(parts[5]) if len(parts) > 5 else job.confidence_threshold
                                        class_name = project_classes.get(class_id, f"class_{class_id}")

                                        prediction = AutoLabelPrediction(
                                            job_id=job_id, image_id=img.id, class_id=class_id,
                                            class_name=class_name, confidence=confidence,
                                            x_center=x_center, y_center=y_center,
                                            width=width, height=height, status="pending",
                                        )
                                        db.add(prediction)
                                        predictions_count += 1

                except subprocess.TimeoutExpired:
                    print(f"Batch {batch_idx + 1} timed out")
                except Exception as e:
                    print(f"Error processing batch {batch_idx + 1}: {e}")

                # Update progress after each batch
                job.processed_images = base_processed + batch_end
                job.predictions_count = predictions_count
                db.commit()

                # Cleanup batch directories
                if batch_dir.exists():
                    shutil.rmtree(batch_dir)
                if output_dir.exists():
                    shutil.rmtree(output_dir)

        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()

        # Final cleanup
        for temp_path in [
            Path(f"/tmp/autolabel/job_{job_id}"),
            Path(f"/tmp/autolabel/job_{job_id}_batch"),
            Path(f"/tmp/autolabel/job_{job_id}_out"),
        ]:
            if temp_path.exists():
                shutil.rmtree(temp_path)

    except Exception as e:
        import traceback
        job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = f"{str(e)}\n{traceback.format_exc()}"
            job.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


# ============ Helper Functions ============

def _update_project_counts(project_id: int, db: Session):
    """Update project image and annotation counts"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        return

    total = db.query(AnnotationImage).filter(AnnotationImage.project_id == project_id).count()
    annotated = db.query(AnnotationImage).filter(
        AnnotationImage.project_id == project_id,
        AnnotationImage.is_annotated == True
    ).count()

    project.total_images = total
    project.annotated_images = annotated
    project.updated_at = datetime.utcnow()
    db.commit()


# ============ VLM Auto-Labeling ============

class VLMJobCreate(BaseModel):
    """Request model for creating a VLM auto-labeling job"""
    project_id: int
    provider: str  # "anthropic", "openai", "ollama"
    classes: List[str]  # Classes to detect
    confidence_threshold: float = 0.5
    batch_size: int = 10  # Smaller batches for API rate limits
    only_unannotated: bool = True
    custom_prompt: Optional[str] = None


def _get_vlm_settings(db: Session) -> dict:
    """Get VLM-related settings from database"""
    from database import SystemSettings

    settings = {}
    keys = [
        "vlm_anthropic_api_key",
        "vlm_openai_api_key",
        "vlm_ollama_endpoint",
        "vlm_ollama_model"
    ]

    for key in keys:
        setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
        if setting:
            settings[key] = setting.value

    # Default values - use OLLAMA_HOST env var if set (for Docker), otherwise localhost
    import os
    if "vlm_ollama_endpoint" not in settings:
        settings["vlm_ollama_endpoint"] = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if "vlm_ollama_model" not in settings:
        settings["vlm_ollama_model"] = "llava:13b"

    return settings


@router.get("/vlm/providers")
async def get_vlm_providers(db: Session = Depends(get_db)):
    """Get list of available VLM providers and their status"""
    from api.vlm_providers import OllamaProvider

    settings = _get_vlm_settings(db)
    providers = []

    # Anthropic Claude
    anthropic_key = settings.get("vlm_anthropic_api_key")
    providers.append({
        "name": "anthropic",
        "display_name": "Anthropic Claude",
        "provider_type": "cloud",
        "is_configured": bool(anthropic_key),
        "is_available": bool(anthropic_key),
        "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
        "default_model": "claude-sonnet-4-20250514",
        "estimated_cost_per_image": 0.01
    })

    # OpenAI GPT-4V
    openai_key = settings.get("vlm_openai_api_key")
    providers.append({
        "name": "openai",
        "display_name": "OpenAI GPT-4 Vision",
        "provider_type": "cloud",
        "is_configured": bool(openai_key),
        "is_available": bool(openai_key),
        "models": ["gpt-4o", "gpt-4-turbo"],
        "default_model": "gpt-4o",
        "estimated_cost_per_image": 0.008
    })

    # Ollama (local)
    ollama_endpoint = settings.get("vlm_ollama_endpoint", "http://localhost:11434")
    ollama_available = False
    ollama_models = []

    try:
        ollama_provider = OllamaProvider(endpoint=ollama_endpoint)
        ollama_available = ollama_provider.validate_connection()
        if ollama_available:
            ollama_models = ollama_provider.list_available_models()
    except Exception:
        pass

    providers.append({
        "name": "ollama",
        "display_name": "Ollama (Local)",
        "provider_type": "local",
        "is_configured": True,  # Always configured with default endpoint
        "is_available": ollama_available,
        "models": ollama_models if ollama_models else ["llava:13b", "llava:7b"],
        "default_model": settings.get("vlm_ollama_model", "llava:13b"),
        "estimated_cost_per_image": 0.0,
        "endpoint": ollama_endpoint
    })

    return {"providers": providers}


@router.get("/vlm/cost-estimate")
async def estimate_vlm_cost(
    project_id: int,
    provider: str,
    only_unannotated: bool = True,
    db: Session = Depends(get_db)
):
    """Estimate cost for VLM auto-labeling"""
    from api.vlm_providers import get_vlm_provider

    settings = _get_vlm_settings(db)

    try:
        vlm_provider = get_vlm_provider(provider, settings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if only_unannotated:
        image_count = db.query(AnnotationImage).filter(
            AnnotationImage.project_id == project_id,
            AnnotationImage.is_annotated == False
        ).count()
    else:
        image_count = db.query(AnnotationImage).filter(
            AnnotationImage.project_id == project_id
        ).count()

    return {
        "image_count": image_count,
        "estimated_cost": vlm_provider.get_cost_estimate(image_count),
        "provider": provider,
        "is_free": provider == "ollama"
    }


@router.post("/vlm/jobs")
async def create_vlm_job(
    job_data: VLMJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create and start a VLM auto-labeling job"""
    from api.vlm_providers import get_vlm_provider

    # Verify project exists
    project = db.query(AnnotationProject).filter(AnnotationProject.id == job_data.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate provider
    settings = _get_vlm_settings(db)
    try:
        provider = get_vlm_provider(job_data.provider, settings)
        if not provider.validate_connection():
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{job_data.provider}' is not available. Check configuration in Settings."
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Count images
    if job_data.only_unannotated:
        image_count = db.query(AnnotationImage).filter(
            AnnotationImage.project_id == job_data.project_id,
            AnnotationImage.is_annotated == False
        ).count()
    else:
        image_count = db.query(AnnotationImage).filter(
            AnnotationImage.project_id == job_data.project_id
        ).count()

    if image_count == 0:
        raise HTTPException(status_code=400, detail="No images to process")

    # Validate classes
    if not job_data.classes or len(job_data.classes) == 0:
        raise HTTPException(status_code=400, detail="At least one class must be specified")

    # Estimate cost
    estimated_cost = provider.get_cost_estimate(image_count)

    # Create job record
    job = AutoLabelJob(
        project_id=job_data.project_id,
        model_path="",  # Not used for VLM
        model_name=f"VLM: {job_data.provider}",
        model_type="vlm",
        vlm_provider=job_data.provider,
        vlm_classes=job_data.classes,
        vlm_prompt=job_data.custom_prompt,
        confidence_threshold=job_data.confidence_threshold,
        batch_size=job_data.batch_size,
        only_unannotated=job_data.only_unannotated,
        estimated_cost=estimated_cost,
        actual_cost=0.0,
        status="pending",
        total_images=image_count,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start background task
    background_tasks.add_task(run_vlm_autolabel_job, job.id)

    return {
        "id": job.id,
        "status": "pending",
        "total_images": image_count,
        "estimated_cost": estimated_cost,
        "provider": job_data.provider,
        "classes": job_data.classes,
        "message": "VLM auto-labeling job started"
    }


def run_vlm_autolabel_job(job_id: int):
    """Run VLM inference on images for auto-labeling (background task)"""
    import asyncio
    from database import SessionLocal
    from api.vlm_providers import get_vlm_provider, detect_with_retry

    db = SessionLocal()
    try:
        job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
        if not job:
            return

        job.status = "running"
        if not job.started_at:
            job.started_at = datetime.utcnow()
        db.commit()

        # Get provider
        settings = _get_vlm_settings(db)
        provider = get_vlm_provider(job.vlm_provider, settings)
        classes = job.vlm_classes or []

        # Get images
        if job.only_unannotated:
            images = db.query(AnnotationImage).filter(
                AnnotationImage.project_id == job.project_id,
                AnnotationImage.is_annotated == False
            ).all()
        else:
            images = db.query(AnnotationImage).filter(
                AnnotationImage.project_id == job.project_id
            ).all()

        predictions_count = 0
        total_cost = 0.0

        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            for idx, image in enumerate(images):
                # Check for pause
                if _paused_jobs.get(job_id):
                    print(f"VLM Job {job_id} paused at image {idx}/{len(images)}")
                    job.status = "paused"
                    db.commit()
                    return

                try:
                    image_path = STORAGE_PATH / image.file_path
                    if not image_path.exists():
                        print(f"Image not found: {image_path}")
                        continue

                    # Run VLM inference with retry
                    response = loop.run_until_complete(
                        detect_with_retry(provider, image_path, classes, job.vlm_prompt)
                    )

                    if response.error:
                        print(f"VLM error on image {image.id}: {response.error}")
                        # Continue to next image, don't fail the whole job
                        continue

                    total_cost += response.cost

                    # Get project classes for mapping
                    project_classes = {c.name.lower(): c.class_index for c in
                        db.query(AnnotationClass).filter(AnnotationClass.project_id == job.project_id).all()}

                    # Store predictions
                    for bbox in response.bboxes:
                        if bbox.confidence < job.confidence_threshold:
                            continue

                        # Try to map to project class (case-insensitive)
                        class_id = project_classes.get(bbox.class_name.lower(), 0)

                        prediction = AutoLabelPrediction(
                            job_id=job_id,
                            image_id=image.id,
                            class_id=class_id,
                            class_name=bbox.class_name,
                            confidence=bbox.confidence,
                            x_center=bbox.x_center,
                            y_center=bbox.y_center,
                            width=bbox.width,
                            height=bbox.height,
                            status="pending",
                        )
                        db.add(prediction)
                        predictions_count += 1

                    # Update progress
                    job.processed_images = idx + 1
                    job.predictions_count = predictions_count
                    job.actual_cost = total_cost
                    db.commit()

                    # Rate limiting for cloud APIs
                    if job.vlm_provider in ["anthropic", "openai"]:
                        import time
                        time.sleep(0.5)  # ~120 RPM

                except Exception as e:
                    print(f"Error processing image {image.id}: {e}")
                    continue

        finally:
            loop.close()

        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.actual_cost = total_cost
        db.commit()

        print(f"VLM Job {job_id} completed: {predictions_count} predictions, ${total_cost:.4f} cost")

    except Exception as e:
        import traceback
        job = db.query(AutoLabelJob).filter(AutoLabelJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = f"{str(e)}\n{traceback.format_exc()}"
            job.completed_at = datetime.utcnow()
            db.commit()
        print(f"VLM Job {job_id} failed: {e}")
    finally:
        db.close()
