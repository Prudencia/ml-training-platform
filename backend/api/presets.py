from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from database import get_db, Preset

router = APIRouter()

class PresetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]  # Training parameters

class PresetUpdate(BaseModel):
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class PresetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    config: Dict[str, Any]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

@router.get("/", response_model=List[PresetResponse])
async def list_presets(db: Session = Depends(get_db)):
    """List all training presets"""
    presets = db.query(Preset).all()
    return [PresetResponse(
        id=p.id,
        name=p.name,
        description=p.description,
        config=p.config,
        created_at=p.created_at.isoformat(),
        updated_at=p.updated_at.isoformat()
    ) for p in presets]

@router.post("/")
async def create_preset(preset_data: PresetCreate, db: Session = Depends(get_db)):
    """Create a new training preset"""
    # Check if preset name already exists
    existing = db.query(Preset).filter(Preset.name == preset_data.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Preset '{preset_data.name}' already exists")

    new_preset = Preset(
        name=preset_data.name,
        description=preset_data.description,
        config=preset_data.config
    )
    db.add(new_preset)
    db.commit()
    db.refresh(new_preset)

    return {"message": "Preset created successfully", "preset_id": new_preset.id}

@router.get("/{preset_id}")
async def get_preset(preset_id: int, db: Session = Depends(get_db)):
    """Get preset details"""
    preset = db.query(Preset).filter(Preset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return preset

@router.put("/{preset_id}")
async def update_preset(preset_id: int, preset_update: PresetUpdate, db: Session = Depends(get_db)):
    """Update a preset"""
    preset = db.query(Preset).filter(Preset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    if preset_update.description is not None:
        preset.description = preset_update.description
    if preset_update.config is not None:
        preset.config = preset_update.config

    db.commit()
    return {"message": "Preset updated successfully"}

@router.delete("/{preset_id}")
async def delete_preset(preset_id: int, db: Session = Depends(get_db)):
    """Delete a preset"""
    preset = db.query(Preset).filter(Preset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    db.delete(preset)
    db.commit()
    return {"message": "Preset deleted successfully"}

@router.get("/defaults/yolov5")
async def get_yolov5_defaults():
    """Get default YOLOv5 training presets"""
    return {
        "small": {
            "img_size": 640,
            "batch_size": 16,
            "epochs": 100,
            "weights": "yolov5s.pt",
            "device": "0"
        },
        "medium": {
            "img_size": 640,
            "batch_size": 8,
            "epochs": 150,
            "weights": "yolov5m.pt",
            "device": "0"
        },
        "large": {
            "img_size": 1280,
            "batch_size": 4,
            "epochs": 300,
            "weights": "yolov5l.pt",
            "device": "0"
        }
    }
