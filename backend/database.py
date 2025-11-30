from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, JSON, Boolean, Float, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
import enum

DATABASE_URL = "sqlite:///./database/mlplatform.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Enums
class UserRole(enum.Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


# System Settings Model (for optional features like auth)
class SystemSettings(Base):
    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    value = Column(Text)
    value_type = Column(String, default="string")  # string, bool, int, json
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# User Model (for optional multi-user support)
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    password_hash = Column(String)
    role = Column(String, default="user")  # admin, user, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    projects = relationship("AnnotationProject", back_populates="owner")


# Auto-Label Job Model
class AutoLabelJob(Base):
    __tablename__ = "auto_label_jobs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("annotation_projects.id"), index=True)
    model_path = Column(String)  # Path to weights file
    model_name = Column(String)  # Display name (e.g., "yolov5s.pt")
    confidence_threshold = Column(Float, default=0.25)
    batch_size = Column(Integer, default=1000)  # Number of images to process per batch
    only_unannotated = Column(Boolean, default=True)  # If True, only process unannotated images
    status = Column(String, default="pending")  # pending, running, completed, failed
    total_images = Column(Integer, default=0)
    processed_images = Column(Integer, default=0)
    predictions_count = Column(Integer, default=0)
    approved_count = Column(Integer, default=0)
    rejected_count = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    project = relationship("AnnotationProject", back_populates="auto_label_jobs")


# Auto-Label Prediction Model (stores predictions before approval)
class AutoLabelPrediction(Base):
    __tablename__ = "auto_label_predictions"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("auto_label_jobs.id"), index=True)
    image_id = Column(Integer, ForeignKey("annotation_images.id"), index=True)
    class_id = Column(Integer)
    class_name = Column(String)
    confidence = Column(Float)
    x_center = Column(Float)
    y_center = Column(Float)
    width = Column(Float)
    height = Column(Float)
    status = Column(String, default="pending")  # pending, approved, rejected
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    job = relationship("AutoLabelJob")
    image = relationship("AnnotationImage")

# Database Models
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    path = Column(String)
    size_bytes = Column(Integer)
    format = Column(String)  # yolov5, coco, etc.
    yaml_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    description = Column(Text, nullable=True)
    num_images = Column(Integer, nullable=True)
    num_classes = Column(Integer, nullable=True)

class VirtualEnvironment(Base):
    __tablename__ = "virtual_environments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    path = Column(String)
    python_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    github_repo = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    venv_id = Column(Integer)
    dataset_id = Column(Integer)
    config_path = Column(String)
    status = Column(String)  # pending, running, paused, completed, failed, queued
    pid = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer)
    base_epochs = Column(Integer, default=0)  # Epochs from previous training (for continued training)
    metrics = Column(JSON, nullable=True)  # Store latest metrics
    log_path = Column(String)
    model_output_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    # Training parameters for resume
    batch_size = Column(Integer, nullable=True)
    img_size = Column(Integer, nullable=True)
    weights = Column(String, nullable=True)
    device = Column(String, nullable=True)
    # Queue management
    queue_position = Column(Integer, nullable=True)
    scheduled_at = Column(DateTime, nullable=True)
    priority = Column(Integer, default=0)  # Higher = more priority

class Preset(Base):
    __tablename__ = "presets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    config = Column(JSON)  # Store all training parameters
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class YAMLConfig(Base):
    __tablename__ = "yaml_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    file_path = Column(String)
    content = Column(Text)
    config_type = Column(String)  # dataset, model, training
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Export(Base):
    __tablename__ = "exports"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, index=True)
    job_name = Column(String)
    status = Column(String)  # pending, running, completed, failed
    format = Column(String)  # tflite, onnx, etc.
    img_size = Column(Integer)
    output_path = Column(String, nullable=True)
    file_size_mb = Column(Float, nullable=True)
    log_path = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    venv_id = Column(Integer, nullable=True)  # Which venv was used for export
    # Training metrics at time of export (snapshot)
    metrics_epochs = Column(Integer, nullable=True)
    metrics_precision = Column(Float, nullable=True)
    metrics_recall = Column(Float, nullable=True)
    metrics_map50 = Column(Float, nullable=True)
    metrics_map50_95 = Column(Float, nullable=True)

class DetectXBuild(Base):
    __tablename__ = "detectx_builds"

    id = Column(Integer, primary_key=True, index=True)
    export_id = Column(Integer, index=True)
    acap_name = Column(String)
    friendly_name = Column(String)
    version = Column(String)
    platform = Column(String)  # A8, A9, TPU
    image_size = Column(Integer)
    vendor = Column(String)
    status = Column(String)  # pending, running, completed, failed
    output_path = Column(String, nullable=True)
    file_size_mb = Column(Float, nullable=True)
    log_path = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)


# Annotation Models
class AnnotationProject(Base):
    __tablename__ = "annotation_projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String, default="active")  # active, exported

    # Owner (optional - only used when auth is enabled)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Configuration stored as JSON
    preprocessing_config = Column(JSON, default=dict)
    augmentation_config = Column(JSON, default=dict)

    # Split ratios for export
    train_ratio = Column(Float, default=0.7)
    val_ratio = Column(Float, default=0.2)
    test_ratio = Column(Float, default=0.1)

    # Progress tracking
    total_images = Column(Integer, default=0)
    annotated_images = Column(Integer, default=0)

    # Relationships
    owner = relationship("User", back_populates="projects")
    classes = relationship("AnnotationClass", back_populates="project", cascade="all, delete-orphan")
    images = relationship("AnnotationImage", back_populates="project", cascade="all, delete-orphan")
    auto_label_jobs = relationship("AutoLabelJob", back_populates="project", cascade="all, delete-orphan")


class AnnotationClass(Base):
    __tablename__ = "annotation_classes"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("annotation_projects.id"), index=True)
    class_index = Column(Integer)  # 0-indexed for YOLO format
    name = Column(String)
    color = Column(String, default="#EF4444")  # Default red

    # Relationship
    project = relationship("AnnotationProject", back_populates="classes")


class AnnotationImage(Base):
    __tablename__ = "annotation_images"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("annotation_projects.id"), index=True)
    source_dataset_id = Column(Integer, nullable=True)  # If imported from existing dataset
    filename = Column(String)  # Unique filename in storage
    original_filename = Column(String, nullable=True)  # Original uploaded filename
    file_path = Column(String)  # Relative path within project storage
    original_width = Column(Integer)
    original_height = Column(Integer)
    is_annotated = Column(Boolean, default=False)
    annotation_count = Column(Integer, default=0)
    split = Column(String, nullable=True)  # train/val/test (assigned during export)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("AnnotationProject", back_populates="images")
    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("annotation_images.id"), index=True)
    class_id = Column(Integer)  # References AnnotationClass.class_index
    # YOLO format: normalized coordinates (0-1)
    x_center = Column(Float)
    y_center = Column(Float)
    width = Column(Float)
    height = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    image = relationship("AnnotationImage", back_populates="annotations")


def init_db():
    """Initialize database and create tables"""
    os.makedirs("database", exist_ok=True)
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
