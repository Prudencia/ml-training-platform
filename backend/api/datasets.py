from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from pathlib import Path
import shutil
import os
import json

from database import get_db, Dataset
from pydantic import BaseModel

router = APIRouter()

STORAGE_PATH = Path("storage/datasets")
STORAGE_PATH.mkdir(parents=True, exist_ok=True)

class DatasetResponse(BaseModel):
    id: int
    name: str
    path: str
    size_bytes: int
    format: str
    yaml_path: Optional[str]
    created_at: str
    description: Optional[str]
    num_images: Optional[int]
    num_classes: Optional[int]

    class Config:
        from_attributes = True

@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(db: Session = Depends(get_db)):
    """List all datasets"""
    datasets = db.query(Dataset).all()
    return [DatasetResponse(
        id=d.id,
        name=d.name,
        path=d.path,
        size_bytes=d.size_bytes,
        format=d.format,
        yaml_path=d.yaml_path,
        created_at=d.created_at.isoformat(),
        description=d.description,
        num_images=d.num_images,
        num_classes=d.num_classes
    ) for d in datasets]

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    format: str = Form("yolov5"),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload a dataset (zip file)"""
    # Check if dataset name already exists
    existing = db.query(Dataset).filter(Dataset.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Dataset '{name}' already exists")

    # Create dataset directory
    dataset_path = STORAGE_PATH / name
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    file_path = dataset_path / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(file_path)

    # If it's a zip file, extract it
    if file.filename.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        os.remove(file_path)  # Remove zip after extraction

    # Auto-fix relative paths in data.yaml to absolute paths
    yaml_file = dataset_path / "data.yaml"
    if yaml_file.exists():
        try:
            import yaml
            with open(yaml_file, 'r') as f:
                yaml_content = yaml.safe_load(f)

            if yaml_content:
                modified = False
                # Get absolute path of dataset directory
                abs_dataset_path = dataset_path.resolve()

                # Fix train/val/test paths if they're relative
                for key in ['train', 'val', 'test']:
                    if key in yaml_content and isinstance(yaml_content[key], str):
                        path_val = yaml_content[key]
                        # Check if it's a relative path (starts with ../ or doesn't start with /)
                        if path_val.startswith('../') or (not path_val.startswith('/') and not path_val.startswith('~')):
                            # Convert relative path to absolute
                            # Handle ../train/images -> dataset_path/train/images
                            if path_val.startswith('../'):
                                # Remove ../ prefix and resolve
                                clean_path = path_val.lstrip('../')
                                abs_path = str(abs_dataset_path / clean_path)
                            else:
                                abs_path = str(abs_dataset_path / path_val)
                            yaml_content[key] = abs_path
                            modified = True

                if modified:
                    with open(yaml_file, 'w') as f:
                        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            # Log but don't fail if yaml processing fails
            print(f"Warning: Could not auto-fix data.yaml paths: {e}")

    # Create database entry
    new_dataset = Dataset(
        name=name,
        path=str(dataset_path),
        size_bytes=file_size,
        format=format,
        description=description
    )
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)

    # Auto-analyze dataset to count images and classes
    try:
        import yaml
        # Count images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        num_images = sum(1 for f in dataset_path.rglob('*') if f.suffix.lower() in image_extensions)
        new_dataset.num_images = num_images

        # Try to detect classes from data.yaml
        yaml_file = dataset_path / "data.yaml"
        if yaml_file.exists():
            with open(yaml_file, 'r') as f:
                yaml_content = yaml.safe_load(f)
                if yaml_content:
                    if 'nc' in yaml_content:
                        new_dataset.num_classes = yaml_content['nc']
                    elif 'names' in yaml_content:
                        new_dataset.num_classes = len(yaml_content['names'])
            new_dataset.yaml_path = str(yaml_file)

        db.commit()
    except Exception as e:
        print(f"Warning: Auto-analysis failed: {e}")

    return {"message": "Dataset uploaded successfully", "dataset_id": new_dataset.id}

@router.get("/{dataset_id}")
async def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Get dataset details"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@router.get("/{dataset_id}/browse")
async def browse_dataset(dataset_id: int, path: str = "", db: Session = Depends(get_db)):
    """Browse dataset files and folders"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    full_path = Path(dataset.path) / path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    items = []
    for item in full_path.iterdir():
        items.append({
            "name": item.name,
            "path": str(item.relative_to(dataset.path)),
            "type": "directory" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else None
        })

    return {"items": items, "current_path": path}

@router.get("/{dataset_id}/file")
async def get_dataset_file(dataset_id: int, path: str, db: Session = Depends(get_db)):
    """Serve a file from the dataset"""
    from fastapi.responses import FileResponse

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    full_path = Path(dataset.path) / path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type based on extension
    ext = full_path.suffix.lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.txt': 'text/plain',
    }
    media_type = media_types.get(ext, 'application/octet-stream')

    return FileResponse(full_path, media_type=media_type)


@router.get("/{dataset_id}/preview")
async def get_dataset_image_preview(dataset_id: int, path: str, db: Session = Depends(get_db)):
    """Get image preview data with annotations for dataset browser"""
    import cv2

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    image_path = Path(dataset.path) / path
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    # Get image dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    height, width = img.shape[:2]

    # Try to find corresponding label file (YOLO format)
    # Pattern: images/train/img.jpg -> labels/train/img.txt
    annotations = []
    path_str = str(path)

    # Try different label path patterns
    label_patterns = []
    if '/images/' in path_str:
        label_patterns.append(Path(dataset.path) / (path_str.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'))
    if path_str.startswith('images/'):
        label_patterns.append(Path(dataset.path) / (path_str.replace('images/', 'labels/', 1).rsplit('.', 1)[0] + '.txt'))
    label_patterns.append(image_path.with_suffix('.txt'))
    label_patterns.append(Path(dataset.path) / 'labels' / (image_path.stem + '.txt'))

    label_path = None
    for pattern in label_patterns:
        if pattern.exists():
            label_path = pattern
            break

    # Parse YOLO format labels
    class_names = {}
    # Try to load class names from data.yaml
    yaml_path = Path(dataset.path) / 'data.yaml'
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f)
                names = yaml_data.get('names', [])
                if isinstance(names, list):
                    class_names = {i: name for i, name in enumerate(names)}
                elif isinstance(names, dict):
                    class_names = {int(k): v for k, v in names.items()}
        except:
            pass

    if label_path and label_path.exists():
        try:
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])
                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_names.get(class_id, f'class_{class_id}'),
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': bbox_width,
                            'height': bbox_height,
                        })
        except:
            pass

    return {
        'width': width,
        'height': height,
        'annotations': annotations,
        'label_path': str(label_path) if label_path else None,
    }


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Delete a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete files
    dataset_path = Path(dataset.path)
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    # Delete database entry
    db.delete(dataset)
    db.commit()

    return {"message": "Dataset deleted successfully"}

@router.post("/{dataset_id}/analyze")
async def analyze_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Analyze dataset structure and count images/classes"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = Path(dataset.path)

    # Count images (common image extensions)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    num_images = sum(1 for f in dataset_path.rglob('*') if f.suffix.lower() in image_extensions)

    # Try to detect classes (look for classes.txt or names in yaml)
    num_classes = None
    classes_file = dataset_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            num_classes = len(f.readlines())

    # Also check data.yaml for nc (number of classes)
    if num_classes is None:
        yaml_file = dataset_path / "data.yaml"
        if yaml_file.exists():
            try:
                import yaml
                with open(yaml_file, 'r') as f:
                    yaml_content = yaml.safe_load(f)
                    if yaml_content and 'nc' in yaml_content:
                        num_classes = yaml_content['nc']
                    elif yaml_content and 'names' in yaml_content:
                        num_classes = len(yaml_content['names'])
            except Exception:
                pass

    # Update dataset
    dataset.num_images = num_images
    dataset.num_classes = num_classes
    db.commit()

    return {
        "num_images": num_images,
        "num_classes": num_classes,
        "message": "Dataset analyzed successfully"
    }

class YAMLContent(BaseModel):
    content: str

@router.get("/{dataset_id}/yaml")
async def get_dataset_yaml(dataset_id: int, db: Session = Depends(get_db)):
    """Get dataset YAML configuration"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = Path(dataset.path)
    yaml_file = dataset_path / "data.yaml"

    if yaml_file.exists():
        with open(yaml_file, 'r') as f:
            content = f.read()
    else:
        # Generate default YAML
        content = f"""# YOLOv5 Dataset Configuration
path: ../datasets/{dataset.name}
train: images/train
val: images/val

nc: 1  # number of classes
names: ['class1']  # class names
"""

    return {"content": content, "path": str(yaml_file)}

@router.put("/{dataset_id}/yaml")
async def update_dataset_yaml(dataset_id: int, yaml_data: YAMLContent, db: Session = Depends(get_db)):
    """Update dataset YAML configuration"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = Path(dataset.path)
    yaml_file = dataset_path / "data.yaml"

    # Write YAML content
    with open(yaml_file, 'w') as f:
        f.write(yaml_data.content)

    # Update dataset yaml_path
    dataset.yaml_path = str(yaml_file)
    db.commit()

    return {"message": "YAML saved successfully", "path": str(yaml_file)}


# ============ Import Dataset from URL ============

class URLImportRequest(BaseModel):
    url: str
    name: str
    description: Optional[str] = None
    format: str = "yolov5"  # yolov5, coco


@router.post("/import/url")
async def import_dataset_from_url(
    request: URLImportRequest,
    db: Session = Depends(get_db)
):
    """Import a dataset from a URL (zip file)"""
    import urllib.request
    import tempfile
    import zipfile

    # Check if dataset name already exists
    existing = db.query(Dataset).filter(Dataset.name == request.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Dataset '{request.name}' already exists")

    # Create dataset directory
    dataset_path = STORAGE_PATH / request.name
    dataset_path.mkdir(parents=True, exist_ok=True)

    try:
        # Download file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        urllib.request.urlretrieve(request.url, tmp_path)
        file_size = os.path.getsize(tmp_path)

        # Extract zip
        if request.url.endswith('.zip') or zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            os.remove(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="URL must point to a zip file")

        # Convert COCO format if needed
        if request.format == "coco":
            _convert_coco_to_yolo(dataset_path)

        # Auto-fix relative paths in data.yaml
        _fix_yaml_paths(dataset_path)

        # Create database entry
        new_dataset = Dataset(
            name=request.name,
            path=str(dataset_path),
            size_bytes=file_size,
            format=request.format,
            description=request.description or f"Imported from {request.url}"
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        # Auto-analyze
        _analyze_dataset(new_dataset, dataset_path, db)

        return {"message": "Dataset imported successfully", "dataset_id": new_dataset.id}

    except urllib.error.URLError as e:
        shutil.rmtree(dataset_path, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to download: {str(e)}")
    except Exception as e:
        shutil.rmtree(dataset_path, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


# ============ Import COCO Format Dataset ============

class COCOImportRequest(BaseModel):
    name: str
    description: Optional[str] = None


@router.post("/import/coco")
async def import_coco_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Import a COCO format dataset and convert to YOLO format"""
    import zipfile

    # Check if dataset name already exists
    existing = db.query(Dataset).filter(Dataset.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Dataset '{name}' already exists")

    # Create dataset directory
    dataset_path = STORAGE_PATH / name
    dataset_path.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploaded file
        file_path = dataset_path / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)

        # Extract if it's a zip
        if file.filename.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            os.remove(file_path)

        # Convert COCO to YOLO
        _convert_coco_to_yolo(dataset_path)

        # Create database entry
        new_dataset = Dataset(
            name=name,
            path=str(dataset_path),
            size_bytes=file_size,
            format="yolov5",  # Converted to YOLO
            description=description or "Imported from COCO format"
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        # Auto-analyze
        _analyze_dataset(new_dataset, dataset_path, db)

        return {"message": "COCO dataset imported and converted to YOLO format", "dataset_id": new_dataset.id}

    except Exception as e:
        shutil.rmtree(dataset_path, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


# ============ Helper Functions ============

def _convert_coco_to_yolo(dataset_path: Path):
    """Convert COCO format annotations to YOLO format"""
    import yaml

    # Find COCO annotation files
    coco_files = list(dataset_path.rglob("*.json"))

    for coco_file in coco_files:
        # Skip if not a COCO annotation file
        if not any(name in coco_file.name.lower() for name in ['instances', 'annotations', 'train', 'val', 'test']):
            continue

        try:
            with open(coco_file) as f:
                coco_data = json.load(f)

            # Check if it's a valid COCO format
            if 'images' not in coco_data or 'annotations' not in coco_data:
                continue

            # Build category mapping
            categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
            category_id_to_yolo = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}

            # Determine split from filename
            split = 'train'
            if 'val' in coco_file.stem.lower():
                split = 'val'
            elif 'test' in coco_file.stem.lower():
                split = 'test'

            # Create output directories
            images_dir = dataset_path / "images" / split
            labels_dir = dataset_path / "labels" / split
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Build image id to filename mapping
            image_info = {img['id']: img for img in coco_data['images']}

            # Group annotations by image
            from collections import defaultdict
            image_annotations = defaultdict(list)
            for ann in coco_data['annotations']:
                image_annotations[ann['image_id']].append(ann)

            # Find source images directory
            possible_dirs = [
                coco_file.parent / "images",
                coco_file.parent / split,
                dataset_path / "images",
                dataset_path / split,
                coco_file.parent,
            ]

            source_images_dir = None
            for d in possible_dirs:
                if d.exists() and any(d.iterdir()):
                    source_images_dir = d
                    break

            # Process each image
            for img_id, img_data in image_info.items():
                img_filename = img_data['file_name']
                img_width = img_data['width']
                img_height = img_data['height']

                # Find and copy image if needed
                if source_images_dir:
                    src_img = None
                    for possible_src in [
                        source_images_dir / img_filename,
                        source_images_dir / Path(img_filename).name,
                    ]:
                        if possible_src.exists():
                            src_img = possible_src
                            break

                    if src_img:
                        dest_img = images_dir / Path(img_filename).name
                        if not dest_img.exists():
                            shutil.copy2(src_img, dest_img)

                # Create YOLO label file
                label_filename = Path(img_filename).stem + ".txt"
                label_path = labels_dir / label_filename

                annotations = image_annotations.get(img_id, [])
                yolo_lines = []

                for ann in annotations:
                    if 'bbox' not in ann:
                        continue

                    # COCO bbox format: [x, y, width, height] (top-left)
                    x, y, w, h = ann['bbox']

                    # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    norm_w = w / img_width
                    norm_h = h / img_height

                    # Get YOLO class index
                    yolo_class = category_id_to_yolo.get(ann['category_id'], 0)

                    yolo_lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))

            # Create data.yaml
            class_names = [categories[cat_id] for cat_id in sorted(categories.keys())]
            yaml_content = {
                "path": str(dataset_path.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(class_names),
                "names": class_names
            }

            yaml_path = dataset_path / "data.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)

        except Exception as e:
            print(f"Warning: Could not convert {coco_file}: {e}")
            continue


def _fix_yaml_paths(dataset_path: Path):
    """Fix relative paths in data.yaml to absolute paths"""
    import yaml

    yaml_file = dataset_path / "data.yaml"
    if not yaml_file.exists():
        return

    try:
        with open(yaml_file, 'r') as f:
            yaml_content = yaml.safe_load(f)

        if not yaml_content:
            return

        modified = False
        abs_dataset_path = dataset_path.resolve()

        for key in ['train', 'val', 'test']:
            if key in yaml_content and isinstance(yaml_content[key], str):
                path_val = yaml_content[key]
                if path_val.startswith('../') or (not path_val.startswith('/') and not path_val.startswith('~')):
                    if path_val.startswith('../'):
                        clean_path = path_val.lstrip('../')
                        abs_path = str(abs_dataset_path / clean_path)
                    else:
                        abs_path = str(abs_dataset_path / path_val)
                    yaml_content[key] = abs_path
                    modified = True

        if modified:
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Warning: Could not fix data.yaml paths: {e}")


def _analyze_dataset(dataset: Dataset, dataset_path: Path, db: Session):
    """Analyze dataset to count images and classes"""
    import yaml

    try:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        num_images = sum(1 for f in dataset_path.rglob('*') if f.suffix.lower() in image_extensions)
        dataset.num_images = num_images

        yaml_file = dataset_path / "data.yaml"
        if yaml_file.exists():
            with open(yaml_file, 'r') as f:
                yaml_content = yaml.safe_load(f)
                if yaml_content:
                    if 'nc' in yaml_content:
                        dataset.num_classes = yaml_content['nc']
                    elif 'names' in yaml_content:
                        dataset.num_classes = len(yaml_content['names'])
            dataset.yaml_path = str(yaml_file)

        db.commit()
    except Exception as e:
        print(f"Warning: Auto-analysis failed: {e}")
