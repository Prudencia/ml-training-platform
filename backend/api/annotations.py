from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
import shutil
import uuid
import os
import json

from PIL import Image, ImageOps
import yaml
import cv2
import numpy as np
import albumentations as A
import random

from database import (
    get_db,
    AnnotationProject,
    AnnotationClass,
    AnnotationImage,
    Annotation,
    Dataset
)

router = APIRouter()

# Storage paths
STORAGE_PATH = Path("storage")
ANNOTATION_PROJECTS_PATH = STORAGE_PATH / "annotation_projects"
ANNOTATION_PROJECTS_PATH.mkdir(parents=True, exist_ok=True)

# Class colors for auto-assignment
CLASS_COLORS = [
    "#EF4444",  # red
    "#F59E0B",  # amber
    "#10B981",  # emerald
    "#3B82F6",  # blue
    "#8B5CF6",  # violet
    "#EC4899",  # pink
    "#06B6D4",  # cyan
    "#84CC16",  # lime
    "#F97316",  # orange
    "#6366F1",  # indigo
]


# ============ Image Processing Helpers ============

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def letterbox_image(img: np.ndarray, target_size: int, color: tuple = (0, 0, 0)) -> tuple:
    """
    Resize image with letterboxing to maintain aspect ratio.
    Returns: (processed_image, scale, pad_x, pad_y)
    """
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas with padding color
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)

    # Center the image
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    return canvas, scale, pad_x, pad_y

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess_image(img: np.ndarray, config: dict) -> tuple:
    """
    Apply preprocessing to image based on config.
    Returns: (processed_image, transform_info)
    """
    transform_info = {"scale": 1.0, "pad_x": 0, "pad_y": 0, "original_size": img.shape[:2]}

    # Auto-orient is handled by PIL before conversion to cv2

    # Apply CLAHE if enabled
    if config.get("clahe_enabled", False):
        clip_limit = config.get("clahe_clip_limit", 2.0)
        img = apply_clahe(img, clip_limit)

    # Resize with letterbox if enabled
    if config.get("resize_enabled", False):
        target_size = config.get("target_size", 640)
        letterbox = config.get("letterbox", True)
        color = hex_to_rgb(config.get("letterbox_color", "#000000"))

        if letterbox:
            img, scale, pad_x, pad_y = letterbox_image(img, target_size, color)
            transform_info["scale"] = scale
            transform_info["pad_x"] = pad_x
            transform_info["pad_y"] = pad_y
        else:
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            h, w = transform_info["original_size"]
            transform_info["scale_x"] = target_size / w
            transform_info["scale_y"] = target_size / h

    transform_info["new_size"] = img.shape[:2]
    return img, transform_info

def transform_bbox(bbox: dict, transform_info: dict, letterbox: bool = True) -> dict:
    """
    Transform bounding box coordinates after preprocessing.
    Input bbox format: {class_id, x_center, y_center, width, height} (normalized 0-1)
    """
    orig_h, orig_w = transform_info["original_size"]
    new_h, new_w = transform_info["new_size"]

    # Denormalize to original pixel coords
    x_center = bbox["x_center"] * orig_w
    y_center = bbox["y_center"] * orig_h
    width = bbox["width"] * orig_w
    height = bbox["height"] * orig_h

    if letterbox:
        scale = transform_info["scale"]
        pad_x = transform_info["pad_x"]
        pad_y = transform_info["pad_y"]

        # Scale and add padding offset
        x_center = x_center * scale + pad_x
        y_center = y_center * scale + pad_y
        width = width * scale
        height = height * scale
    else:
        scale_x = transform_info.get("scale_x", 1.0)
        scale_y = transform_info.get("scale_y", 1.0)
        x_center = x_center * scale_x
        y_center = y_center * scale_y
        width = width * scale_x
        height = height * scale_y

    # Normalize to new image size
    return {
        "class_id": bbox["class_id"],
        "x_center": x_center / new_w,
        "y_center": y_center / new_h,
        "width": width / new_w,
        "height": height / new_h
    }

def get_augmentation_pipeline(config: dict) -> A.Compose:
    """Create albumentations augmentation pipeline based on config"""
    transforms = []

    if config.get("horizontal_flip", False):
        transforms.append(A.HorizontalFlip(p=0.5))

    if config.get("vertical_flip", False):
        transforms.append(A.VerticalFlip(p=0.5))

    rotation = config.get("rotation_range", 0)
    if rotation > 0:
        transforms.append(A.Rotate(limit=rotation, p=0.7, border_mode=cv2.BORDER_CONSTANT))

    brightness = config.get("brightness_range", 0)
    contrast = config.get("contrast_range", 0)
    if brightness > 0 or contrast > 0:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=brightness,
            contrast_limit=contrast,
            p=0.7
        ))

    saturation = config.get("saturation_range", 0)
    if saturation > 0:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=int(saturation * 100),
            val_shift_limit=0,
            p=0.5
        ))

    if config.get("noise_enabled", False):
        transforms.append(A.GaussNoise(var_limit=(10, 50), p=0.3))

    if config.get("blur_enabled", False):
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))

    return A.Compose(transforms, bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))


# ============ Pydantic Models ============

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    initial_classes: Optional[List[str]] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    augmentation_config: Optional[Dict[str, Any]] = None
    train_ratio: Optional[float] = None
    val_ratio: Optional[float] = None
    test_ratio: Optional[float] = None

class ClassCreate(BaseModel):
    name: str
    color: Optional[str] = None

class ClassUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None

class AnnotationCreate(BaseModel):
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

class AnnotationBatchSave(BaseModel):
    annotations: List[AnnotationCreate]

class ImportRequest(BaseModel):
    dataset_ids: List[int]
    include_annotations: bool = True

class SplitConfig(BaseModel):
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    method: str = "random"  # random or stratified
    seed: Optional[int] = 42

class ExportConfig(BaseModel):
    name: str
    description: Optional[str] = None
    apply_augmentation: bool = False
    augmentation_copies: int = 2


# ============ Project Endpoints ============

@router.get("/projects")
async def list_projects(db: Session = Depends(get_db)):
    """List all annotation projects"""
    projects = db.query(AnnotationProject).order_by(AnnotationProject.updated_at.desc()).all()

    result = []
    for p in projects:
        # Get class count
        class_count = db.query(AnnotationClass).filter(AnnotationClass.project_id == p.id).count()
        # Get total annotation count
        total_annotations = db.query(func.sum(AnnotationImage.annotation_count)).filter(
            AnnotationImage.project_id == p.id
        ).scalar() or 0

        result.append({
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "status": p.status,
            "total_images": p.total_images,
            "annotated_images": p.annotated_images,
            "total_annotations": total_annotations,
            "class_count": class_count,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "updated_at": p.updated_at.isoformat() if p.updated_at else None,
        })

    return result


@router.post("/projects")
async def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new annotation project"""
    # Check if name exists
    existing = db.query(AnnotationProject).filter(AnnotationProject.name == project.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Project with this name already exists")

    # Create project
    new_project = AnnotationProject(
        name=project.name,
        description=project.description,
        preprocessing_config={},
        augmentation_config={},
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    # Create project folder
    project_path = ANNOTATION_PROJECTS_PATH / str(new_project.id)
    (project_path / "images").mkdir(parents=True, exist_ok=True)
    (project_path / "thumbnails").mkdir(parents=True, exist_ok=True)

    # Add initial classes if provided
    if project.initial_classes:
        for idx, class_name in enumerate(project.initial_classes):
            new_class = AnnotationClass(
                project_id=new_project.id,
                class_index=idx,
                name=class_name,
                color=CLASS_COLORS[idx % len(CLASS_COLORS)]
            )
            db.add(new_class)
        db.commit()

    return {"message": "Project created", "id": new_project.id}


@router.get("/projects/{project_id}")
async def get_project(project_id: int, db: Session = Depends(get_db)):
    """Get project details"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get classes
    classes = db.query(AnnotationClass).filter(
        AnnotationClass.project_id == project_id
    ).order_by(AnnotationClass.class_index).all()

    # Get total annotation count
    total_annotations = db.query(func.sum(AnnotationImage.annotation_count)).filter(
        AnnotationImage.project_id == project_id
    ).scalar() or 0

    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "status": project.status,
        "total_images": project.total_images,
        "annotated_images": project.annotated_images,
        "total_annotations": total_annotations,
        "preprocessing_config": project.preprocessing_config or {},
        "augmentation_config": project.augmentation_config or {},
        "train_ratio": project.train_ratio,
        "val_ratio": project.val_ratio,
        "test_ratio": project.test_ratio,
        "classes": [
            {
                "id": c.id,
                "class_index": c.class_index,
                "name": c.name,
                "color": c.color
            } for c in classes
        ],
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
    }


@router.put("/projects/{project_id}")
async def update_project(project_id: int, update: ProjectUpdate, db: Session = Depends(get_db)):
    """Update project settings"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if update.name is not None:
        # Check if name is taken by another project
        existing = db.query(AnnotationProject).filter(
            AnnotationProject.name == update.name,
            AnnotationProject.id != project_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Name already taken")
        project.name = update.name

    if update.description is not None:
        project.description = update.description
    if update.preprocessing_config is not None:
        project.preprocessing_config = update.preprocessing_config
    if update.augmentation_config is not None:
        project.augmentation_config = update.augmentation_config
    if update.train_ratio is not None:
        project.train_ratio = update.train_ratio
    if update.val_ratio is not None:
        project.val_ratio = update.val_ratio
    if update.test_ratio is not None:
        project.test_ratio = update.test_ratio

    project.updated_at = datetime.utcnow()
    db.commit()

    return {"message": "Project updated"}


@router.delete("/projects/{project_id}")
async def delete_project(project_id: int, db: Session = Depends(get_db)):
    """Delete a project and all its data"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Delete project folder
    project_path = ANNOTATION_PROJECTS_PATH / str(project_id)
    if project_path.exists():
        shutil.rmtree(project_path)

    # Delete from database (cascade will delete classes, images, annotations)
    db.delete(project)
    db.commit()

    return {"message": "Project deleted"}


# ============ Class Endpoints ============

@router.get("/projects/{project_id}/classes")
async def list_classes(project_id: int, db: Session = Depends(get_db)):
    """List all classes in a project"""
    classes = db.query(AnnotationClass).filter(
        AnnotationClass.project_id == project_id
    ).order_by(AnnotationClass.class_index).all()

    result = []
    for c in classes:
        # Count annotations for this class
        count = db.query(Annotation).join(AnnotationImage).filter(
            AnnotationImage.project_id == project_id,
            Annotation.class_id == c.class_index
        ).count()

        result.append({
            "id": c.id,
            "class_index": c.class_index,
            "name": c.name,
            "color": c.color,
            "annotation_count": count
        })

    return result


@router.post("/projects/{project_id}/classes")
async def create_class(project_id: int, cls: ClassCreate, db: Session = Depends(get_db)):
    """Add a new class to the project"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get next class index
    max_index = db.query(func.max(AnnotationClass.class_index)).filter(
        AnnotationClass.project_id == project_id
    ).scalar()
    # Note: Use "is None" check because max_index could be 0 (falsy but valid)
    next_index = 0 if max_index is None else max_index + 1

    # Auto-assign color if not provided
    color = cls.color or CLASS_COLORS[next_index % len(CLASS_COLORS)]

    new_class = AnnotationClass(
        project_id=project_id,
        class_index=next_index,
        name=cls.name,
        color=color
    )
    db.add(new_class)
    db.commit()
    db.refresh(new_class)

    return {
        "id": new_class.id,
        "class_index": new_class.class_index,
        "name": new_class.name,
        "color": new_class.color
    }


@router.put("/projects/{project_id}/classes/{class_id}")
async def update_class(project_id: int, class_id: int, update: ClassUpdate, db: Session = Depends(get_db)):
    """Update class name or color"""
    cls = db.query(AnnotationClass).filter(
        AnnotationClass.id == class_id,
        AnnotationClass.project_id == project_id
    ).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    if update.name is not None:
        cls.name = update.name
    if update.color is not None:
        cls.color = update.color

    db.commit()
    return {"message": "Class updated"}


@router.delete("/projects/{project_id}/classes/{class_id}")
async def delete_class(
    project_id: int,
    class_id: int,
    delete_annotations: bool = Query(True, description="Also delete annotations with this class"),
    db: Session = Depends(get_db)
):
    """Delete a class and optionally its annotations"""
    cls = db.query(AnnotationClass).filter(
        AnnotationClass.id == class_id,
        AnnotationClass.project_id == project_id
    ).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    class_index = cls.class_index

    if delete_annotations:
        # Delete all annotations with this class
        image_ids = db.query(AnnotationImage.id).filter(
            AnnotationImage.project_id == project_id
        ).all()
        image_ids = [i[0] for i in image_ids]

        db.query(Annotation).filter(
            Annotation.image_id.in_(image_ids),
            Annotation.class_id == class_index
        ).delete(synchronize_session=False)

        # Update annotation counts
        for img_id in image_ids:
            count = db.query(Annotation).filter(Annotation.image_id == img_id).count()
            db.query(AnnotationImage).filter(AnnotationImage.id == img_id).update({
                "annotation_count": count,
                "is_annotated": count > 0
            })

    db.delete(cls)
    db.commit()

    # Update project counts
    _update_project_counts(project_id, db)

    return {"message": "Class deleted"}


class BulkRemapRequest(BaseModel):
    mappings: dict  # {old_class_id: new_class_id}


@router.get("/projects/{project_id}/annotation-class-stats")
async def get_annotation_class_stats(project_id: int, db: Session = Depends(get_db)):
    """Get statistics of class IDs used in annotations (including orphaned ones not in project classes)"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get all image IDs for this project
    image_ids = db.query(AnnotationImage.id).filter(AnnotationImage.project_id == project_id).all()
    image_ids = [i[0] for i in image_ids]

    if not image_ids:
        return {"classes": [], "project_classes": []}

    # Get unique class_ids and counts from annotations
    from sqlalchemy import func
    results = db.query(
        Annotation.class_id,
        func.count(Annotation.id).label('count')
    ).filter(
        Annotation.image_id.in_(image_ids)
    ).group_by(Annotation.class_id).all()

    # Get project classes for reference
    project_classes = db.query(AnnotationClass).filter(
        AnnotationClass.project_id == project_id
    ).order_by(AnnotationClass.class_index).all()

    class_map = {c.class_index: c for c in project_classes}

    annotation_classes = []
    for class_id, count in results:
        cls = class_map.get(class_id)
        annotation_classes.append({
            "class_id": class_id,
            "count": count,
            "name": cls.name if cls else f"Unknown (ID: {class_id})",
            "color": cls.color if cls else "#888888",
            "is_orphaned": cls is None
        })

    return {
        "classes": sorted(annotation_classes, key=lambda x: x["class_id"]),
        "project_classes": [
            {"class_index": c.class_index, "name": c.name, "color": c.color, "id": c.id}
            for c in project_classes
        ]
    }


@router.post("/projects/{project_id}/bulk-remap-annotations")
async def bulk_remap_annotations(project_id: int, request: BulkRemapRequest, db: Session = Depends(get_db)):
    """Bulk remap annotation class IDs. mappings = {old_class_id: new_class_id}"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get all image IDs for this project
    image_ids = db.query(AnnotationImage.id).filter(AnnotationImage.project_id == project_id).all()
    image_ids = [i[0] for i in image_ids]

    if not image_ids:
        return {"message": "No images in project", "updated_count": 0}

    total_updated = 0
    for old_class_id, new_class_id in request.mappings.items():
        old_id = int(old_class_id)
        new_id = int(new_class_id)

        if old_id == new_id:
            continue

        updated = db.query(Annotation).filter(
            Annotation.image_id.in_(image_ids),
            Annotation.class_id == old_id
        ).update({"class_id": new_id}, synchronize_session=False)

        total_updated += updated

    db.commit()

    return {"message": f"Updated {total_updated} annotations", "updated_count": total_updated}


# ============ Image Endpoints ============

@router.get("/projects/{project_id}/images")
async def list_images(
    project_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=50000),
    filter_status: Optional[str] = Query(None, description="Filter by annotation status: annotated, unannotated"),
    has_class: Optional[int] = Query(None, description="Filter to images that have annotations of this class_index"),
    missing_class: Optional[int] = Query(None, description="Filter to images missing annotations of this class_index"),
    db: Session = Depends(get_db)
):
    """List images in project with pagination"""
    query = db.query(AnnotationImage).filter(AnnotationImage.project_id == project_id)

    if filter_status == "annotated":
        query = query.filter(AnnotationImage.is_annotated == True)
    elif filter_status == "unannotated":
        query = query.filter(AnnotationImage.is_annotated == False)

    # Filter by class presence
    if has_class is not None:
        # Images that have at least one annotation of this class
        subquery = db.query(Annotation.image_id).filter(Annotation.class_id == has_class).distinct()
        query = query.filter(AnnotationImage.id.in_(subquery))
    elif missing_class is not None:
        # Images that don't have any annotations of this class
        subquery = db.query(Annotation.image_id).filter(Annotation.class_id == missing_class).distinct()
        query = query.filter(~AnnotationImage.id.in_(subquery))

    total = query.count()
    images = query.order_by(AnnotationImage.id).offset((page - 1) * per_page).limit(per_page).all()

    return {
        "images": [
            {
                "id": img.id,
                "filename": img.filename,
                "original_width": img.original_width,
                "original_height": img.original_height,
                "is_annotated": img.is_annotated,
                "annotation_count": img.annotation_count,
                "split": img.split,
            } for img in images
        ],
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page
    }


@router.post("/projects/{project_id}/images/upload")
async def upload_images(
    project_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload images to a project"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_path = ANNOTATION_PROJECTS_PATH / str(project_id)
    images_path = project_path / "images"
    thumbnails_path = project_path / "thumbnails"

    uploaded = 0
    errors = []

    for file in files:
        try:
            # Check if it's an image
            if not file.content_type or not file.content_type.startswith("image/"):
                errors.append(f"{file.filename}: Not an image")
                continue

            # Generate unique filename
            ext = Path(file.filename).suffix.lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                errors.append(f"{file.filename}: Unsupported format")
                continue

            unique_name = f"{uuid.uuid4().hex}{ext}"
            file_path = images_path / unique_name

            # Save file
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Get image dimensions
            with Image.open(file_path) as img:
                width, height = img.size

                # Create thumbnail (max 256px)
                thumb_size = (256, 256)
                img.thumbnail(thumb_size, Image.LANCZOS)
                thumb_path = thumbnails_path / f"{unique_name}"
                img.save(thumb_path, quality=85)

            # Create database record
            new_image = AnnotationImage(
                project_id=project_id,
                filename=unique_name,
                original_filename=file.filename,  # Store original filename
                file_path=str(file_path.relative_to(STORAGE_PATH)),
                original_width=width,
                original_height=height,
            )
            db.add(new_image)
            uploaded += 1

        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    # Update project count
    if uploaded > 0:
        project.total_images += uploaded
        project.updated_at = datetime.utcnow()
        db.commit()

    return {
        "uploaded": uploaded,
        "errors": errors
    }


@router.post("/projects/{project_id}/images/import")
async def import_from_datasets(
    project_id: int,
    request: ImportRequest,
    db: Session = Depends(get_db)
):
    """Import images from existing datasets"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_path = ANNOTATION_PROJECTS_PATH / str(project_id)
    images_path = project_path / "images"
    thumbnails_path = project_path / "thumbnails"

    # Get existing classes for mapping
    existing_classes = {c.name.lower(): c.class_index for c in project.classes}

    imported = 0
    annotations_imported = 0
    errors = []

    for dataset_id in request.dataset_ids:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            errors.append(f"Dataset {dataset_id} not found")
            continue

        dataset_path = Path(dataset.path)
        if not dataset_path.exists():
            errors.append(f"Dataset path not found: {dataset_path}")
            continue

        # Load class names from data.yaml if exists
        yaml_path = dataset_path / "data.yaml"
        class_names = {}
        if yaml_path.exists():
            try:
                with open(yaml_path) as f:
                    yaml_data = yaml.safe_load(f)
                    if "names" in yaml_data:
                        if isinstance(yaml_data["names"], dict):
                            class_names = {int(k): v for k, v in yaml_data["names"].items()}
                        elif isinstance(yaml_data["names"], list):
                            class_names = {i: name for i, name in enumerate(yaml_data["names"])}
            except Exception as e:
                errors.append(f"Error reading data.yaml: {str(e)}")

        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        for img_file in dataset_path.rglob("*"):
            if img_file.suffix.lower() not in image_extensions:
                continue

            # Skip if in labels directory
            if "labels" in str(img_file):
                continue

            try:
                # Generate unique filename
                unique_name = f"{uuid.uuid4().hex}{img_file.suffix.lower()}"
                dest_path = images_path / unique_name

                # Copy image
                shutil.copy2(img_file, dest_path)

                # Get dimensions and create thumbnail
                with Image.open(dest_path) as img:
                    width, height = img.size
                    img.thumbnail((256, 256), Image.LANCZOS)
                    img.save(thumbnails_path / unique_name, quality=85)

                # Create image record
                new_image = AnnotationImage(
                    project_id=project_id,
                    source_dataset_id=dataset_id,
                    filename=unique_name,
                    original_filename=img_file.name,  # Store original filename
                    file_path=str(dest_path.relative_to(STORAGE_PATH)),
                    original_width=width,
                    original_height=height,
                )
                db.add(new_image)
                db.flush()  # Get the ID

                imported += 1

                # Import annotations if requested
                if request.include_annotations:
                    # Try to find corresponding label file
                    label_path = None

                    # Try different label path patterns for YOLO datasets
                    relative = img_file.relative_to(dataset_path)

                    # Handle YOLO structure: train/images/img.jpg -> train/labels/img.txt
                    # Replace 'images' folder with 'labels' in path
                    relative_str = str(relative)
                    if '/images/' in relative_str or relative_str.startswith('images/'):
                        labels_relative = relative_str.replace('/images/', '/labels/').replace('images/', 'labels/', 1)
                        labels_path = dataset_path / Path(labels_relative).with_suffix(".txt")
                    else:
                        labels_path = None

                    for pattern in [
                        # YOLO structure: train/images/ -> train/labels/
                        labels_path,
                        # Labels in dataset root
                        dataset_path / "labels" / relative.with_suffix(".txt"),
                        # Label next to image
                        img_file.with_suffix(".txt"),
                        # Labels in subfolder matching image subfolder
                        dataset_path / "labels" / relative.parent.name / (relative.stem + ".txt"),
                    ]:
                        if pattern and pattern.exists():
                            label_path = pattern
                            break

                    if label_path and label_path.exists():
                        image_ann_count = 0  # Track annotations for this image locally
                        with open(label_path) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    try:
                                        orig_class_id = int(parts[0])
                                        x_center = float(parts[1])
                                        y_center = float(parts[2])
                                        bbox_width = float(parts[3])
                                        bbox_height = float(parts[4])

                                        # Map class to project class
                                        orig_class_name = class_names.get(orig_class_id, f"class_{orig_class_id}")

                                        # Check if class exists in project, if not create it
                                        if orig_class_name.lower() not in existing_classes:
                                            next_index = max(existing_classes.values(), default=-1) + 1
                                            new_class = AnnotationClass(
                                                project_id=project_id,
                                                class_index=next_index,
                                                name=orig_class_name,
                                                color=CLASS_COLORS[next_index % len(CLASS_COLORS)]
                                            )
                                            db.add(new_class)
                                            db.flush()
                                            existing_classes[orig_class_name.lower()] = next_index

                                        project_class_id = existing_classes[orig_class_name.lower()]

                                        # Create annotation
                                        new_annotation = Annotation(
                                            image_id=new_image.id,
                                            class_id=project_class_id,
                                            x_center=x_center,
                                            y_center=y_center,
                                            width=bbox_width,
                                            height=bbox_height,
                                        )
                                        db.add(new_annotation)
                                        annotations_imported += 1
                                        image_ann_count += 1

                                    except (ValueError, IndexError):
                                        continue

                        # Update image annotation count using local counter
                        new_image.annotation_count = image_ann_count
                        new_image.is_annotated = image_ann_count > 0

            except Exception as e:
                errors.append(f"{img_file.name}: {str(e)}")

    # Update project counts
    _update_project_counts(project_id, db)
    db.commit()

    return {
        "imported_images": imported,
        "imported_annotations": annotations_imported,
        "errors": errors
    }


@router.get("/projects/{project_id}/images/{image_id}/file")
async def serve_image(
    project_id: int,
    image_id: int,
    max_size: int = None,
    db: Session = Depends(get_db)
):
    """Serve an image file, optionally scaled down for viewing"""
    from fastapi.responses import StreamingResponse
    import io

    image = db.query(AnnotationImage).filter(
        AnnotationImage.id == image_id,
        AnnotationImage.project_id == project_id
    ).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = STORAGE_PATH / image.file_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    # If max_size specified and image is larger, serve scaled version (no file caching)
    if max_size and max_size > 0:
        try:
            img = cv2.imread(str(file_path))
            if img is not None:
                h, w = img.shape[:2]
                if max(w, h) > max_size:
                    # Scale down
                    scale = max_size / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Encode to JPEG in memory and stream it
                    _, buffer = cv2.imencode('.jpg', scaled, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    return StreamingResponse(
                        io.BytesIO(buffer.tobytes()),
                        media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"}  # Browser cache for 24h
                    )
        except Exception as e:
            # Fall back to full image on error
            pass

    return FileResponse(
        file_path,
        headers={"Cache-Control": "public, max-age=31536000"}
    )


@router.get("/projects/{project_id}/images/{image_id}/thumbnail")
async def serve_thumbnail(project_id: int, image_id: int, db: Session = Depends(get_db)):
    """Serve an image thumbnail"""
    image = db.query(AnnotationImage).filter(
        AnnotationImage.id == image_id,
        AnnotationImage.project_id == project_id
    ).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    thumb_path = ANNOTATION_PROJECTS_PATH / str(project_id) / "thumbnails" / image.filename
    if not thumb_path.exists():
        # Fall back to full image
        return await serve_image(project_id, image_id, db)

    return FileResponse(thumb_path)


@router.delete("/projects/{project_id}/images/{image_id}")
async def delete_image(project_id: int, image_id: int, db: Session = Depends(get_db)):
    """Delete an image and its annotations"""
    image = db.query(AnnotationImage).filter(
        AnnotationImage.id == image_id,
        AnnotationImage.project_id == project_id
    ).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete files
    file_path = STORAGE_PATH / image.file_path
    if file_path.exists():
        file_path.unlink()

    thumb_path = ANNOTATION_PROJECTS_PATH / str(project_id) / "thumbnails" / image.filename
    if thumb_path.exists():
        thumb_path.unlink()

    # Delete from database (cascade will delete annotations)
    db.delete(image)
    db.commit()

    # Update project counts
    _update_project_counts(project_id, db)

    return {"message": "Image deleted"}


@router.get("/projects/{project_id}/next-unannotated")
async def get_next_unannotated(project_id: int, db: Session = Depends(get_db)):
    """Get the next unannotated image for resume functionality"""
    # First try to find unannotated image
    next_image = db.query(AnnotationImage).filter(
        AnnotationImage.project_id == project_id,
        AnnotationImage.is_annotated == False
    ).order_by(AnnotationImage.id).first()

    if next_image:
        return {"image_id": next_image.id, "type": "unannotated"}

    # If all annotated, return first image
    first_image = db.query(AnnotationImage).filter(
        AnnotationImage.project_id == project_id
    ).order_by(AnnotationImage.id).first()

    if first_image:
        return {"image_id": first_image.id, "type": "first"}

    return {"image_id": None, "type": "empty"}


# ============ Annotation Endpoints ============

@router.get("/images/{image_id}/annotations")
async def get_annotations(image_id: int, db: Session = Depends(get_db)):
    """Get all annotations for an image"""
    image = db.query(AnnotationImage).filter(AnnotationImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    annotations = db.query(Annotation).filter(Annotation.image_id == image_id).all()

    return {
        "image_id": image_id,
        "original_width": image.original_width,
        "original_height": image.original_height,
        "annotations": [
            {
                "id": a.id,
                "class_id": a.class_id,
                "x_center": a.x_center,
                "y_center": a.y_center,
                "width": a.width,
                "height": a.height,
            } for a in annotations
        ]
    }


@router.post("/images/{image_id}/annotations")
async def save_annotations(image_id: int, data: AnnotationBatchSave, db: Session = Depends(get_db)):
    """Save all annotations for an image (batch replace)"""
    image = db.query(AnnotationImage).filter(AnnotationImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete existing annotations
    db.query(Annotation).filter(Annotation.image_id == image_id).delete()

    # Add new annotations
    for ann in data.annotations:
        new_annotation = Annotation(
            image_id=image_id,
            class_id=ann.class_id,
            x_center=ann.x_center,
            y_center=ann.y_center,
            width=ann.width,
            height=ann.height,
        )
        db.add(new_annotation)

    # Update image status
    image.annotation_count = len(data.annotations)
    image.is_annotated = len(data.annotations) > 0

    db.commit()

    # Update project counts
    _update_project_counts(image.project_id, db)

    return {"message": "Annotations saved", "count": len(data.annotations)}


@router.delete("/images/{image_id}/annotations")
async def clear_image_annotations(image_id: int, db: Session = Depends(get_db)):
    """Clear all annotations from an image"""
    image = db.query(AnnotationImage).filter(AnnotationImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    db.query(Annotation).filter(Annotation.image_id == image_id).delete()
    image.annotation_count = 0
    image.is_annotated = False
    db.commit()

    # Update project counts
    _update_project_counts(image.project_id, db)

    return {"message": "Annotations cleared"}


@router.delete("/projects/{project_id}/annotations")
async def clear_project_annotations(project_id: int, db: Session = Depends(get_db)):
    """Clear all annotations from entire project"""
    # Get all image IDs
    image_ids = db.query(AnnotationImage.id).filter(
        AnnotationImage.project_id == project_id
    ).all()
    image_ids = [i[0] for i in image_ids]

    # Delete all annotations
    db.query(Annotation).filter(Annotation.image_id.in_(image_ids)).delete(synchronize_session=False)

    # Update all images
    db.query(AnnotationImage).filter(AnnotationImage.project_id == project_id).update({
        "annotation_count": 0,
        "is_annotated": False
    })

    db.commit()

    # Update project counts
    _update_project_counts(project_id, db)

    return {"message": "All annotations cleared"}


@router.delete("/projects/{project_id}/annotations/by-class/{class_id}")
async def delete_annotations_by_class(project_id: int, class_id: int, db: Session = Depends(get_db)):
    """Delete all annotations with a specific class_id (useful for removing orphaned class annotations)"""
    # Get all image IDs for this project
    image_ids = db.query(AnnotationImage.id).filter(
        AnnotationImage.project_id == project_id
    ).all()
    image_ids = [i[0] for i in image_ids]

    if not image_ids:
        raise HTTPException(status_code=404, detail="Project has no images")

    # Delete annotations with this class_id
    deleted_count = db.query(Annotation).filter(
        Annotation.image_id.in_(image_ids),
        Annotation.class_id == class_id
    ).delete(synchronize_session=False)

    # Update annotation counts for affected images
    for img_id in image_ids:
        remaining = db.query(Annotation).filter(Annotation.image_id == img_id).count()
        db.query(AnnotationImage).filter(AnnotationImage.id == img_id).update({
            "annotation_count": remaining,
            "is_annotated": remaining > 0
        })

    db.commit()

    # Update project counts
    _update_project_counts(project_id, db)

    return {"message": f"Deleted {deleted_count} annotations with class_id {class_id}"}


# ============ Split & Export Endpoints ============

@router.post("/projects/{project_id}/generate-splits")
async def generate_splits(project_id: int, config: SplitConfig, db: Session = Depends(get_db)):
    """Generate train/val/test splits"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate ratios
    total = config.train_ratio + config.val_ratio + config.test_ratio
    if abs(total - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Split ratios must sum to 1.0")

    # Get annotated images
    images = db.query(AnnotationImage).filter(
        AnnotationImage.project_id == project_id,
        AnnotationImage.is_annotated == True
    ).all()

    if not images:
        raise HTTPException(status_code=400, detail="No annotated images to split")

    import random
    if config.seed:
        random.seed(config.seed)

    # Shuffle
    image_ids = [img.id for img in images]
    random.shuffle(image_ids)

    n = len(image_ids)
    train_end = int(n * config.train_ratio)
    val_end = train_end + int(n * config.val_ratio)

    train_ids = image_ids[:train_end]
    val_ids = image_ids[train_end:val_end]
    test_ids = image_ids[val_end:]

    # Update splits
    db.query(AnnotationImage).filter(AnnotationImage.id.in_(train_ids)).update(
        {"split": "train"}, synchronize_session=False
    )
    db.query(AnnotationImage).filter(AnnotationImage.id.in_(val_ids)).update(
        {"split": "val"}, synchronize_session=False
    )
    db.query(AnnotationImage).filter(AnnotationImage.id.in_(test_ids)).update(
        {"split": "test"}, synchronize_session=False
    )

    # Save ratios to project
    project.train_ratio = config.train_ratio
    project.val_ratio = config.val_ratio
    project.test_ratio = config.test_ratio

    db.commit()

    return {
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
    }


@router.get("/projects/{project_id}/export/preview")
async def preview_export(project_id: int, db: Session = Depends(get_db)):
    """Preview export statistics"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Count by split
    train_count = db.query(AnnotationImage).filter(
        AnnotationImage.project_id == project_id,
        AnnotationImage.split == "train"
    ).count()

    val_count = db.query(AnnotationImage).filter(
        AnnotationImage.project_id == project_id,
        AnnotationImage.split == "val"
    ).count()

    test_count = db.query(AnnotationImage).filter(
        AnnotationImage.project_id == project_id,
        AnnotationImage.split == "test"
    ).count()

    unsplit_count = db.query(AnnotationImage).filter(
        AnnotationImage.project_id == project_id,
        AnnotationImage.split == None,
        AnnotationImage.is_annotated == True
    ).count()

    # Get classes
    classes = db.query(AnnotationClass).filter(
        AnnotationClass.project_id == project_id
    ).order_by(AnnotationClass.class_index).all()

    return {
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "unsplit_annotated": unsplit_count,
        "classes": [{"index": c.class_index, "name": c.name} for c in classes],
        "preprocessing_config": project.preprocessing_config,
        "augmentation_config": project.augmentation_config,
    }


# ============ Quality Checks Endpoint ============

@router.get("/projects/{project_id}/quality-check")
async def quality_check(project_id: int, db: Session = Depends(get_db)):
    """
    Run annotation quality checks on a project.
    Detects:
    - Images without annotations
    - Very small bounding boxes (< 1% of image area)
    - Very large bounding boxes (> 90% of image area)
    - Overlapping boxes (IoU > 0.8)
    - Truncated boxes at image edges
    """
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    issues_data = {
        "unannotated_images": [],
        "small_boxes": [],
        "large_boxes": [],
        "overlapping_boxes": [],
        "edge_truncated": [],
    }

    # Thresholds
    SMALL_BOX_THRESHOLD = 0.01  # 1% of image area
    LARGE_BOX_THRESHOLD = 0.90  # 90% of image area
    OVERLAP_IOU_THRESHOLD = 0.8
    EDGE_THRESHOLD = 0.02  # 2% from edge

    # Get all images
    images = db.query(AnnotationImage).filter(
        AnnotationImage.project_id == project_id
    ).all()

    # Track stats
    total_annotations = 0
    images_with_annotations = 0

    for image in images:
        # Check for unannotated images
        if not image.is_annotated or image.annotation_count == 0:
            issues_data["unannotated_images"].append({
                "image_id": image.id,
                "filename": image.filename
            })
            continue

        # Get annotations for this image
        annotations = db.query(Annotation).filter(Annotation.image_id == image.id).all()

        if not annotations:
            issues_data["unannotated_images"].append({
                "image_id": image.id,
                "filename": image.filename
            })
            continue

        # Count annotations and images with annotations
        total_annotations += len(annotations)
        images_with_annotations += 1

        boxes_for_overlap_check = []

        for ann in annotations:
            box_area = ann.width * ann.height

            # Check for small boxes
            if box_area < SMALL_BOX_THRESHOLD:
                issues_data["small_boxes"].append({
                    "image_id": image.id,
                    "filename": image.filename,
                    "annotation_id": ann.id,
                    "class_id": ann.class_id,
                    "area_percent": round(box_area * 100, 2)
                })

            # Check for large boxes
            if box_area > LARGE_BOX_THRESHOLD:
                issues_data["large_boxes"].append({
                    "image_id": image.id,
                    "filename": image.filename,
                    "annotation_id": ann.id,
                    "class_id": ann.class_id,
                    "area_percent": round(box_area * 100, 2)
                })

            # Check for edge truncation
            x1 = ann.x_center - ann.width / 2
            y1 = ann.y_center - ann.height / 2
            x2 = ann.x_center + ann.width / 2
            y2 = ann.y_center + ann.height / 2

            if x1 < EDGE_THRESHOLD or y1 < EDGE_THRESHOLD or x2 > (1 - EDGE_THRESHOLD) or y2 > (1 - EDGE_THRESHOLD):
                issues_data["edge_truncated"].append({
                    "image_id": image.id,
                    "filename": image.filename,
                    "annotation_id": ann.id,
                    "class_id": ann.class_id,
                    "edge": "left" if x1 < EDGE_THRESHOLD else "top" if y1 < EDGE_THRESHOLD else "right" if x2 > (1 - EDGE_THRESHOLD) else "bottom"
                })

            # Store box for overlap check
            boxes_for_overlap_check.append({
                "ann": ann,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

        # Check for overlapping boxes
        for i, box1 in enumerate(boxes_for_overlap_check):
            for box2 in boxes_for_overlap_check[i+1:]:
                # Calculate IoU
                xi1 = max(box1["x1"], box2["x1"])
                yi1 = max(box1["y1"], box2["y1"])
                xi2 = min(box1["x2"], box2["x2"])
                yi2 = min(box1["y2"], box2["y2"])

                if xi1 < xi2 and yi1 < yi2:
                    intersection = (xi2 - xi1) * (yi2 - yi1)
                    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
                    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou > OVERLAP_IOU_THRESHOLD:
                        issues_data["overlapping_boxes"].append({
                            "image_id": image.id,
                            "filename": image.filename,
                            "annotation_ids": [box1["ann"].id, box2["ann"].id],
                            "class_ids": [box1["ann"].class_id, box2["ann"].class_id],
                            "iou": round(iou, 2)
                        })

    # Build issues array for frontend (format: [{severity, type, message, count}, ...])
    issues = []

    if issues_data["unannotated_images"]:
        issues.append({
            "severity": "warning",
            "type": "Unannotated Images",
            "message": "Images without any bounding boxes",
            "count": len(issues_data["unannotated_images"])
        })

    if issues_data["small_boxes"]:
        issues.append({
            "severity": "warning",
            "type": "Small Bounding Boxes",
            "message": f"Boxes smaller than {SMALL_BOX_THRESHOLD * 100}% of image area",
            "count": len(issues_data["small_boxes"])
        })

    if issues_data["large_boxes"]:
        issues.append({
            "severity": "warning",
            "type": "Large Bounding Boxes",
            "message": f"Boxes larger than {LARGE_BOX_THRESHOLD * 100}% of image area",
            "count": len(issues_data["large_boxes"])
        })

    if issues_data["overlapping_boxes"]:
        issues.append({
            "severity": "error",
            "type": "Overlapping Boxes",
            "message": f"Boxes with IoU > {OVERLAP_IOU_THRESHOLD * 100}% (possible duplicates)",
            "count": len(issues_data["overlapping_boxes"])
        })

    if issues_data["edge_truncated"]:
        issues.append({
            "severity": "info",
            "type": "Edge-Truncated Boxes",
            "message": "Boxes touching image edges (may be intentional)",
            "count": len(issues_data["edge_truncated"])
        })

    # Calculate health score based on percentage of images without issues
    # Only count significant issues (not edge truncation as info)
    significant_issue_images = set()
    for item in issues_data["unannotated_images"]:
        significant_issue_images.add(item["image_id"])
    for item in issues_data["small_boxes"]:
        significant_issue_images.add(item["image_id"])
    for item in issues_data["large_boxes"]:
        significant_issue_images.add(item["image_id"])
    for item in issues_data["overlapping_boxes"]:
        significant_issue_images.add(item["image_id"])

    images_with_issues = len(significant_issue_images)
    health_score = round(100 * (1 - (images_with_issues / max(len(images), 1))), 1)
    health_score = max(0, min(100, health_score))  # Clamp to 0-100

    return {
        "project_id": project_id,
        "total_images": len(images),
        "total_annotations": total_annotations,
        "images_with_annotations": images_with_annotations,
        "total_issues": sum(len(v) for v in issues_data.values()),
        "issues": issues,
        "issues_data": issues_data,  # Keep raw data for debugging/details
        "summary": {
            "unannotated_count": len(issues_data["unannotated_images"]),
            "small_boxes_count": len(issues_data["small_boxes"]),
            "large_boxes_count": len(issues_data["large_boxes"]),
            "overlapping_count": len(issues_data["overlapping_boxes"]),
            "edge_truncated_count": len(issues_data["edge_truncated"]),
        },
        "health_score": health_score
    }


# ============ Quality Fix Endpoints ============

@router.post("/projects/{project_id}/fix/delete-small-boxes")
async def delete_small_boxes(project_id: int, threshold: float = 0.01, db: Session = Depends(get_db)):
    """Delete all bounding boxes smaller than threshold (default 1% of image area)"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get all annotations and delete small ones
    annotations = db.query(Annotation).join(AnnotationImage).filter(
        AnnotationImage.project_id == project_id
    ).all()

    deleted_count = 0
    affected_images = set()

    for ann in annotations:
        box_area = ann.width * ann.height
        if box_area < threshold:
            affected_images.add(ann.image_id)
            db.delete(ann)
            deleted_count += 1

    # Update annotation counts for affected images
    for image_id in affected_images:
        image = db.query(AnnotationImage).filter(AnnotationImage.id == image_id).first()
        if image:
            remaining = db.query(Annotation).filter(Annotation.image_id == image_id).count()
            image.annotation_count = remaining
            image.is_annotated = remaining > 0

    db.commit()
    return {"deleted": deleted_count, "affected_images": len(affected_images)}


@router.post("/projects/{project_id}/fix/delete-large-boxes")
async def delete_large_boxes(project_id: int, threshold: float = 0.90, db: Session = Depends(get_db)):
    """Delete all bounding boxes larger than threshold (default 90% of image area)"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    annotations = db.query(Annotation).join(AnnotationImage).filter(
        AnnotationImage.project_id == project_id
    ).all()

    deleted_count = 0
    affected_images = set()

    for ann in annotations:
        box_area = ann.width * ann.height
        if box_area > threshold:
            affected_images.add(ann.image_id)
            db.delete(ann)
            deleted_count += 1

    # Update annotation counts
    for image_id in affected_images:
        image = db.query(AnnotationImage).filter(AnnotationImage.id == image_id).first()
        if image:
            remaining = db.query(Annotation).filter(Annotation.image_id == image_id).count()
            image.annotation_count = remaining
            image.is_annotated = remaining > 0

    db.commit()
    return {"deleted": deleted_count, "affected_images": len(affected_images)}


@router.post("/projects/{project_id}/fix/merge-overlapping")
async def merge_overlapping_boxes(project_id: int, iou_threshold: float = 0.8, db: Session = Depends(get_db)):
    """Merge overlapping boxes (IoU > threshold) by keeping the larger one"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    images = db.query(AnnotationImage).filter(AnnotationImage.project_id == project_id).all()

    deleted_count = 0
    affected_images = set()

    for image in images:
        annotations = db.query(Annotation).filter(Annotation.image_id == image.id).all()
        if len(annotations) < 2:
            continue

        # Build boxes list
        boxes = []
        for ann in annotations:
            x1 = ann.x_center - ann.width / 2
            y1 = ann.y_center - ann.height / 2
            x2 = ann.x_center + ann.width / 2
            y2 = ann.y_center + ann.height / 2
            area = ann.width * ann.height
            boxes.append({"ann": ann, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": area})

        # Find overlapping pairs and mark smaller one for deletion
        to_delete = set()
        for i, box1 in enumerate(boxes):
            if box1["ann"].id in to_delete:
                continue
            for j, box2 in enumerate(boxes[i+1:], i+1):
                if box2["ann"].id in to_delete:
                    continue

                # Calculate IoU
                xi1 = max(box1["x1"], box2["x1"])
                yi1 = max(box1["y1"], box2["y1"])
                xi2 = min(box1["x2"], box2["x2"])
                yi2 = min(box1["y2"], box2["y2"])

                if xi1 < xi2 and yi1 < yi2:
                    intersection = (xi2 - xi1) * (yi2 - yi1)
                    union = box1["area"] + box2["area"] - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou > iou_threshold:
                        # Delete the smaller box
                        if box1["area"] >= box2["area"]:
                            to_delete.add(box2["ann"].id)
                        else:
                            to_delete.add(box1["ann"].id)

        # Delete marked annotations
        if to_delete:
            affected_images.add(image.id)
            for ann_id in to_delete:
                ann = db.query(Annotation).filter(Annotation.id == ann_id).first()
                if ann:
                    db.delete(ann)
                    deleted_count += 1

    # Update annotation counts
    for image_id in affected_images:
        image = db.query(AnnotationImage).filter(AnnotationImage.id == image_id).first()
        if image:
            remaining = db.query(Annotation).filter(Annotation.image_id == image_id).count()
            image.annotation_count = remaining
            image.is_annotated = remaining > 0

    db.commit()
    return {"deleted": deleted_count, "affected_images": len(affected_images)}


@router.get("/projects/{project_id}/unannotated-images")
async def get_unannotated_images(project_id: int, db: Session = Depends(get_db)):
    """Get list of images without annotations for quick navigation"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    images = db.query(AnnotationImage).filter(
        AnnotationImage.project_id == project_id,
        (AnnotationImage.is_annotated == False) | (AnnotationImage.annotation_count == 0)
    ).order_by(AnnotationImage.id).limit(100).all()

    return {
        "images": [{"id": img.id, "filename": img.filename} for img in images],
        "total": len(images)
    }


@router.post("/projects/{project_id}/export")
async def export_to_dataset(project_id: int, config: ExportConfig, db: Session = Depends(get_db)):
    """Export annotation project as YOLO-format dataset"""
    project = db.query(AnnotationProject).filter(AnnotationProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check if dataset name exists
    existing = db.query(Dataset).filter(Dataset.name == config.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Dataset with this name already exists")

    # Get classes
    classes = db.query(AnnotationClass).filter(
        AnnotationClass.project_id == project_id
    ).order_by(AnnotationClass.class_index).all()

    if not classes:
        raise HTTPException(status_code=400, detail="No classes defined")

    # Create dataset directory
    dataset_path = STORAGE_PATH / "datasets" / config.name
    for split in ["train", "val", "test"]:
        (dataset_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Get images with splits
    images = db.query(AnnotationImage).filter(
        AnnotationImage.project_id == project_id,
        AnnotationImage.is_annotated == True,
        AnnotationImage.split != None
    ).all()

    if not images:
        # Clean up
        shutil.rmtree(dataset_path)
        raise HTTPException(status_code=400, detail="No images with splits assigned. Generate splits first.")

    exported_images = 0
    exported_annotations = 0

    # Get preprocessing and augmentation configs
    preprocess_config = project.preprocessing_config or {}
    augment_config = project.augmentation_config or {}
    do_augment = augment_config.get("enabled", False)
    augment_copies = augment_config.get("copies", 2) if do_augment else 0

    # Setup augmentation pipeline if needed
    augment_pipeline = get_augmentation_pipeline(augment_config) if do_augment else None

    # Track used filenames to avoid conflicts
    used_filenames = {"train": set(), "val": set(), "test": set()}

    def save_image_and_labels(img_array, bboxes, split, base_name, suffix=""):
        """Helper to save image and corresponding label file"""
        nonlocal exported_images, exported_annotations

        # Generate unique filename
        final_name = f"{base_name}{suffix}" if suffix else base_name
        counter = 1
        while final_name in used_filenames[split]:
            final_name = f"{base_name}{suffix}_{counter}"
            counter += 1
        used_filenames[split].add(final_name)

        # Save image (always as jpg for consistency)
        dest_path = dataset_path / "images" / split / f"{final_name}.jpg"
        cv2.imwrite(str(dest_path), img_array, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save label file
        label_content = []
        for bbox in bboxes:
            label_content.append(f"{int(bbox['class_id'])} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}")

        label_path = dataset_path / "labels" / split / f"{final_name}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(label_content))

        exported_images += 1
        exported_annotations += len(bboxes)

    for img in images:
        split = img.split
        if not split:
            continue

        src_path = STORAGE_PATH / img.file_path
        if not src_path.exists():
            continue

        # Load image with PIL (handles EXIF auto-orient if enabled)
        try:
            pil_img = Image.open(src_path)
            if preprocess_config.get("auto_orient", True):
                pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            continue

        # Get annotations for this image
        annotations = db.query(Annotation).filter(Annotation.image_id == img.id).all()
        bboxes = [
            {"class_id": ann.class_id, "x_center": ann.x_center, "y_center": ann.y_center,
             "width": ann.width, "height": ann.height}
            for ann in annotations
        ]

        # Apply preprocessing
        if preprocess_config.get("resize_enabled") or preprocess_config.get("clahe_enabled"):
            processed_img, transform_info = preprocess_image(cv_img, preprocess_config)

            # Transform bboxes if resizing was applied
            if preprocess_config.get("resize_enabled"):
                letterbox = preprocess_config.get("letterbox", True)
                bboxes = [transform_bbox(bbox, transform_info, letterbox) for bbox in bboxes]
            cv_img = processed_img

        # Use original filename if available
        export_filename = img.original_filename if img.original_filename else img.filename
        base_name = Path(export_filename).stem

        # Save original (preprocessed) image
        save_image_and_labels(cv_img, bboxes, split, base_name)

        # Apply augmentation if enabled (only on training set)
        if do_augment and split == "train" and augment_pipeline and len(bboxes) > 0:
            for aug_idx in range(augment_copies):
                try:
                    # Prepare bboxes for albumentations (YOLO format)
                    yolo_bboxes = [[b["x_center"], b["y_center"], b["width"], b["height"]] for b in bboxes]
                    class_labels = [b["class_id"] for b in bboxes]

                    # Apply augmentation
                    augmented = augment_pipeline(
                        image=cv_img,
                        bboxes=yolo_bboxes,
                        class_labels=class_labels
                    )

                    aug_img = augmented["image"]
                    aug_bboxes = [
                        {"class_id": cl, "x_center": bb[0], "y_center": bb[1], "width": bb[2], "height": bb[3]}
                        for bb, cl in zip(augmented["bboxes"], augmented["class_labels"])
                    ]

                    # Only save if we still have valid bboxes after augmentation
                    if aug_bboxes:
                        save_image_and_labels(aug_img, aug_bboxes, split, base_name, f"_aug{aug_idx+1}")
                except Exception as e:
                    # Skip failed augmentations
                    continue

    # Generate data.yaml
    yaml_content = {
        "path": str(dataset_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": {c.class_index: c.name for c in classes}
    }

    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    # Calculate size
    total_size = sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file())

    # Create Dataset record
    new_dataset = Dataset(
        name=config.name,
        path=str(dataset_path),
        size_bytes=total_size,
        format="yolov5",
        yaml_path=str(yaml_path),
        description=config.description or f"Exported from annotation project: {project.name}",
        num_images=exported_images,
        num_classes=len(classes)
    )
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)

    return {
        "dataset_id": new_dataset.id,
        "dataset_name": config.name,
        "exported_images": exported_images,
        "exported_annotations": exported_annotations,
        "size_bytes": total_size,
    }


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


# ============ Augmentation Preview Endpoints ============

class AugmentationPreviewRequest(BaseModel):
    config: Dict[str, Any]


@router.post("/projects/{project_id}/images/{image_id}/augmentation-preview")
async def get_augmentation_preview(
    project_id: int,
    image_id: int,
    request: AugmentationPreviewRequest,
    db: Session = Depends(get_db)
):
    """Generate augmentation preview for an image"""
    from fastapi.responses import StreamingResponse
    import io
    import base64

    image = db.query(AnnotationImage).filter(
        AnnotationImage.id == image_id,
        AnnotationImage.project_id == project_id
    ).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = STORAGE_PATH / image.file_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    # Load image
    try:
        pil_img = Image.open(file_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load image: {str(e)}")

    # Get annotations
    annotations = db.query(Annotation).filter(Annotation.image_id == image_id).all()
    bboxes = [[ann.x_center, ann.y_center, ann.width, ann.height] for ann in annotations]
    class_labels = [ann.class_id for ann in annotations]

    # Create augmentation pipeline from config
    augment_config = request.config
    pipeline = get_augmentation_pipeline(augment_config)

    # Generate multiple augmented versions
    previews = []

    # Original image
    _, original_buffer = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    previews.append({
        "type": "original",
        "image": base64.b64encode(original_buffer).decode('utf-8'),
        "bboxes": [{"x_center": b[0], "y_center": b[1], "width": b[2], "height": b[3], "class_id": c}
                   for b, c in zip(bboxes, class_labels)]
    })

    # Generate augmented previews
    for i in range(3):  # Generate 3 augmented versions
        try:
            if bboxes:
                augmented = pipeline(
                    image=cv_img,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                aug_img = augmented["image"]
                aug_bboxes = [
                    {"x_center": bb[0], "y_center": bb[1], "width": bb[2], "height": bb[3], "class_id": cl}
                    for bb, cl in zip(augmented["bboxes"], augmented["class_labels"])
                ]
            else:
                # No bboxes, just augment the image
                aug_pipeline = get_augmentation_pipeline(augment_config)
                aug_pipeline = A.Compose([t for t in aug_pipeline.transforms])
                augmented = aug_pipeline(image=cv_img)
                aug_img = augmented["image"]
                aug_bboxes = []

            _, aug_buffer = cv2.imencode('.jpg', aug_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            previews.append({
                "type": f"augmented_{i+1}",
                "image": base64.b64encode(aug_buffer).decode('utf-8'),
                "bboxes": aug_bboxes
            })
        except Exception as e:
            # Skip failed augmentations
            continue

    return {
        "previews": previews,
        "config": augment_config,
        "original_width": image.original_width,
        "original_height": image.original_height,
    }


@router.get("/augmentation-defaults")
async def get_augmentation_defaults():
    """Get default augmentation configuration"""
    return {
        "enabled": False,
        "copies": 2,
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation_range": 15,
        "brightness_range": 0.2,
        "contrast_range": 0.2,
        "saturation_range": 0.2,
        "noise_enabled": False,
        "blur_enabled": False,
    }
