# ML Training Platform

A comprehensive web-based platform for managing machine learning training workflows, with a primary focus on YOLOv5 object detection models and ACAP deployment for Axis cameras. Train models remotely through a browser interface with full control over datasets, annotations, virtual environments, training configurations, model export, and ACAP package generation.

## Features

### Dataset & Annotation Management
- **Dataset Management**: Upload ZIP datasets with drag-and-drop, automatic extraction, and analysis
- **Dataset Browser**: Browse dataset contents with inline image preview and bounding box visualization
- **Annotation Projects**: Full-featured annotation tool for creating and editing YOLO-format labels
  - Bounding box drawing with class assignment
  - Grid overlay and zoom controls
  - Keyboard shortcuts for efficient labeling
  - Import images from datasets or upload directly
  - Auto-split generation (train/val/test)
  - Export to YOLO-format datasets
- **Auto-Labeling**: Run inference with pre-trained models for automatic annotation
  - Upload custom YOLOv5 models or use pre-trained weights
  - Confidence threshold filtering
  - Review, approve, or reject predictions
  - Bulk class remapping for model class alignment
- **Class Management**:
  - Add, edit, rename classes with custom colors
  - Delete classes and all associated annotations
  - Filter images by class (has/missing class)
  - Remap annotations between classes
  - Detect and manage orphaned class annotations

### Training & Virtual Environments
- **Virtual Environment Management**: Create isolated Python environments, clone GitHub repos, manage dependencies
- **Training Job Control**: Start, stop, pause, resume training with real-time progress and live log streaming
- **Training Presets**: Save and reuse configurations (Small/Medium/Large defaults included)
- **Training Queue**: Queue multiple jobs with priority-based execution and scheduling
- **YAML Configuration Editor**: Monaco Editor with syntax highlighting for dataset configs

### Model Export & Deployment
- **TFLite Export**: Convert trained models to TensorFlow Lite format with INT8 quantization
- **Axis YOLOv5 Workflow**: Specialized support for Axis camera-optimized YOLOv5 (ReLU6 activation)
- **DetectX ACAP Builder**: Generate ACAP packages (.eap) for deployment on Axis network cameras (ARTPEC-8/9)

### Monitoring & Tools
- **System Dashboard**: Real-time CPU, GPU, memory, and disk usage monitoring
- **Web Terminal**: Interactive browser-based terminal with full PTY support
- **Support Page**: Donation and contribution links

## Screenshots

| Dashboard | Annotation Tool | Training Jobs |
|-----------|-----------------|---------------|
| System monitoring with GPU stats | Draw bounding boxes with class labels | Real-time training progress |

## Tech Stack

### Backend
- **FastAPI** - High-performance async Python web framework
- **SQLAlchemy** - SQL toolkit and ORM
- **SQLite** - Lightweight file-based database
- **WebSockets** - Real-time log streaming
- **OpenCV** - Image processing for annotations

### Frontend
- **React 18** - Component-based UI library
- **Vite** - Modern JavaScript bundler
- **TailwindCSS** - Utility-first CSS framework
- **Monaco Editor** - VS Code's editor component
- **XTerm.js** - Browser-based terminal
- **Recharts** - Real-time charts
- **Lucide Icons** - Modern icon set

## Project Structure

```
trainplattform/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── database.py             # SQLAlchemy models
│   ├── requirements.txt        # Python dependencies
│   └── api/
│       ├── annotations.py      # Annotation projects, images, labels
│       ├── autolabel.py        # Auto-labeling with pretrained models
│       ├── datasets.py         # Dataset upload, browse, preview
│       ├── training.py         # Training job control
│       ├── venv.py             # Virtual environment management
│       ├── presets.py          # Training presets
│       ├── workflows.py        # Axis YOLOv5, export, DetectX
│       ├── settings.py         # User auth, system settings
│       ├── system.py           # System monitoring
│       ├── terminal.py         # Web terminal
│       └── yaml_config.py      # YAML configuration
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main app with routing
│   │   ├── services/api.js     # Axios API client
│   │   └── pages/
│   │       ├── Dashboard.jsx       # System stats
│   │       ├── Annotate.jsx        # Annotation project list
│   │       ├── AnnotateProject.jsx # Annotation editor
│   │       ├── Datasets.jsx        # Dataset management
│   │       ├── TrainingJobs.jsx    # Training control
│   │       ├── VirtualEnvs.jsx     # Environment management
│   │       ├── Presets.jsx         # Training presets
│   │       ├── Queue.jsx           # Training queue
│   │       ├── Exports.jsx         # Model exports
│   │       ├── AxisYOLOv5.jsx      # Axis workflow
│   │       ├── DetectXBuild.jsx    # ACAP builder
│   │       ├── Terminal.jsx        # Web terminal
│   │       └── Support.jsx         # Donation page
│   └── dist/                   # Production build (generated)
│
└── backend/storage/            # Data storage (auto-created)
    ├── datasets/               # Uploaded datasets
    ├── models/                 # Trained models
    ├── pretrained_models/      # Pre-trained models for auto-labeling
    ├── venvs/                  # Virtual environments
    ├── annotation_projects/    # Annotation project images
    ├── configs/                # YAML configurations
    ├── logs/                   # Training logs
    ├── exports/                # Exported models
    └── detectx_builds/         # ACAP package builds
```

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- Git
- (Optional) NVIDIA GPU with CUDA
- (Optional) Docker for DetectX builds

### 1. Clone and Setup

```bash
git clone https://github.com/Prudencia/ml-training-platform.git
cd ml-training-platform
chmod +x setup.sh && ./setup.sh
```

### 2. Start Backend

```bash
cd backend
source venv/bin/activate
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Start Frontend (Development)

```bash
cd frontend
npm install
npm run dev
```

### 4. Access

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## Workflows

### Complete Training Workflow

```
1. Upload Dataset     → /datasets (ZIP with images/labels)
2. Create Annotation  → /annotate (or skip if already labeled)
   Project
3. Draw Bounding     → /annotate/:id (label objects, assign classes)
   Boxes
4. Generate Splits   → Train/Val/Test split assignment
5. Export Dataset    → Creates YOLO-format dataset
6. Create Venv       → /venvs (clone YOLOv5 repo)
7. Start Training    → /training (configure and run)
8. Monitor Progress  → Real-time logs and metrics
9. Export Model      → /exports (TFLite with INT8)
10. Build ACAP       → /detectx (generate .eap package)
11. Deploy to Camera → Upload .eap to Axis camera
```

### Annotation Workflow

```
1. Create Project    → Define name and initial classes
2. Import Images     → From datasets or upload files/folders
3. Draw Boxes        → Click and drag on images
4. Assign Classes    → Number keys or class panel
5. Navigate          → Arrow keys or thumbnail click
6. Generate Splits   → Auto-assign train/val/test
7. Export            → Creates ready-to-train dataset
```

## Keyboard Shortcuts (Annotation)

| Key | Action |
|-----|--------|
| `1-9` | Select class |
| `Arrow Left/Right` | Previous/Next image |
| `Delete/Backspace` | Delete selected box |
| `Escape` | Cancel drawing |
| `G` | Toggle grid |
| `+/-` | Zoom in/out |

## API Routes

| Route | Description |
|-------|-------------|
| `/api/datasets/` | Dataset CRUD, browse, preview |
| `/api/annotations/` | Annotation projects, images, labels, class management |
| `/api/autolabel/` | Auto-labeling jobs, predictions, model management |
| `/api/training/` | Training job control, logs |
| `/api/venv/` | Virtual environment management |
| `/api/presets/` | Training presets |
| `/api/workflows/` | Axis YOLOv5, export, DetectX |
| `/api/system/` | System monitoring |
| `/api/terminal/` | Web terminal |
| `/api/settings/` | User authentication, settings |

## System Requirements

### Minimum
- CPU: 4 cores
- RAM: 8GB
- Disk: 50GB free
- OS: Linux (Ubuntu 20.04+)

### Recommended
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA 8GB+ VRAM
- Disk: 100GB+ SSD

## Deployment Options

### Option 1: Docker (Recommended)

The easiest way to deploy the platform:

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access at `http://localhost:3000` (frontend) and `http://localhost:8000` (API).

### Option 2: Production (Single Server)

Build the frontend and serve everything from the backend:

```bash
# Build frontend
cd frontend
npm run build
cd ..

# Start backend (serves both API and frontend)
cd backend
source venv/bin/activate
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Access everything at `http://localhost:8000`.

### Option 3: Separate Services

Run backend and frontend separately:

**Backend:**
```bash
cd backend
source venv/bin/activate
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Frontend:**
```bash
cd frontend
npm run build
# Serve with nginx, Apache, or any static file server
```

### Remote Access Configuration

To access the platform from another machine:

1. Edit `frontend/.env`:
   ```
   VITE_API_URL=http://YOUR_SERVER_IP:8000
   ```

2. Rebuild the frontend:
   ```bash
   cd frontend && npm run build
   ```

3. Ensure firewall allows ports 3000 and 8000


## Troubleshooting

### Common Issues

**Training not starting:**
- Check virtual environment has YOLOv5 installed
- Verify dataset path in YAML config
- Check `train.py` exists in venv

**GPU not detected:**
```bash
nvidia-smi  # Verify drivers
# Set device to "0" in training config
```

**Export failing:**
- Ensure model weights file exists
- Check disk space for TFLite export

## Support

If you find this project useful, consider supporting its development:
- Visit the Support page in the app
- Star the repository on GitHub
- Report bugs and suggest features
- Contribute code improvements

## License

MIT License - feel free to use and modify.

## Contributing

Contributions welcome! Please open an issue or PR.

---

**Built for the Axis camera ML deployment workflow**
