from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel
import yaml

from database import get_db, YAMLConfig

router = APIRouter()

CONFIGS_PATH = Path("storage/configs")
CONFIGS_PATH.mkdir(parents=True, exist_ok=True)

class YAMLCreate(BaseModel):
    name: str
    content: str
    config_type: str  # dataset, model, training

class YAMLUpdate(BaseModel):
    content: str

class YAMLResponse(BaseModel):
    id: int
    name: str
    file_path: str
    content: str
    config_type: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

@router.get("/", response_model=List[YAMLResponse])
async def list_yaml_configs(config_type: Optional[str] = None, db: Session = Depends(get_db)):
    """List all YAML configurations"""
    query = db.query(YAMLConfig)
    if config_type:
        query = query.filter(YAMLConfig.config_type == config_type)

    configs = query.all()
    return [YAMLResponse(
        id=c.id,
        name=c.name,
        file_path=c.file_path,
        content=c.content,
        config_type=c.config_type,
        created_at=c.created_at.isoformat(),
        updated_at=c.updated_at.isoformat()
    ) for c in configs]

@router.post("/")
async def create_yaml_config(yaml_data: YAMLCreate, db: Session = Depends(get_db)):
    """Create a new YAML configuration"""
    # Validate YAML syntax
    try:
        yaml.safe_load(yaml_data.content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML syntax: {str(e)}")

    # Check if name already exists
    existing = db.query(YAMLConfig).filter(YAMLConfig.name == yaml_data.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"YAML config '{yaml_data.name}' already exists")

    # Save to file
    file_path = CONFIGS_PATH / f"{yaml_data.name}.yaml"
    with open(file_path, 'w') as f:
        f.write(yaml_data.content)

    # Create database entry
    new_config = YAMLConfig(
        name=yaml_data.name,
        file_path=str(file_path),
        content=yaml_data.content,
        config_type=yaml_data.config_type
    )
    db.add(new_config)
    db.commit()
    db.refresh(new_config)

    return {"message": "YAML config created successfully", "config_id": new_config.id}

@router.get("/{config_id}")
async def get_yaml_config(config_id: int, db: Session = Depends(get_db)):
    """Get YAML configuration details"""
    config = db.query(YAMLConfig).filter(YAMLConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="YAML config not found")
    return config

@router.put("/{config_id}")
async def update_yaml_config(config_id: int, yaml_update: YAMLUpdate, db: Session = Depends(get_db)):
    """Update a YAML configuration"""
    config = db.query(YAMLConfig).filter(YAMLConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="YAML config not found")

    # Validate YAML syntax
    try:
        yaml.safe_load(yaml_update.content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML syntax: {str(e)}")

    # Update file
    with open(config.file_path, 'w') as f:
        f.write(yaml_update.content)

    # Update database
    config.content = yaml_update.content
    db.commit()

    return {"message": "YAML config updated successfully"}

@router.delete("/{config_id}")
async def delete_yaml_config(config_id: int, db: Session = Depends(get_db)):
    """Delete a YAML configuration"""
    config = db.query(YAMLConfig).filter(YAMLConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="YAML config not found")

    # Delete file
    file_path = Path(config.file_path)
    if file_path.exists():
        file_path.unlink()

    # Delete database entry
    db.delete(config)
    db.commit()

    return {"message": "YAML config deleted successfully"}

@router.post("/generate/dataset")
async def generate_dataset_yaml(
    dataset_path: str,
    train_path: str,
    val_path: str,
    nc: int,
    names: List[str],
    db: Session = Depends(get_db)
):
    """Generate a YAML config for a dataset"""
    yaml_content = {
        "path": dataset_path,
        "train": train_path,
        "val": val_path,
        "nc": nc,
        "names": names
    }

    yaml_str = yaml.dump(yaml_content, default_flow_style=False)

    return {"content": yaml_str}

@router.post("/validate")
async def validate_yaml(content: str):
    """Validate YAML syntax"""
    try:
        yaml.safe_load(content)
        return {"valid": True, "message": "YAML is valid"}
    except yaml.YAMLError as e:
        return {"valid": False, "message": str(e)}
