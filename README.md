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
- **Auto-Labeling**: Run inference with pre-trained models or VLMs for automatic annotation
  - Upload custom YOLOv5 models or use pre-trained weights
  - **VLM (Vision Language Model) Support**: Use AI vision models for intelligent auto-labeling
  - Confidence threshold filtering
  - Review, approve, or reject predictions
  - Bulk class remapping for model class alignment

### VLM (Vision Language Models)
- **Local Models via Ollama**: Run vision models locally for free
  - LLaVA (7B, 13B, 34B), LLaMA 3.2 Vision, Moondream, MiniCPM-V, and more
  - One-click model installation from VLM page
  - Custom model support - pull any Ollama vision model
- **Florence-2 (Local)**: Microsoft's vision model with native object detection
  - Runs locally via dedicated venv - no API keys needed
  - Multiple model sizes (Base, Large, Fine-tuned variants)
  - GPU/CPU device selection for flexible inference
- **Cloud Providers**: Use cloud APIs for higher accuracy
  - **NVIDIA NIM**: 1000 free credits for new users (Phi-3.5 Vision, VILA, LLaMA Vision)
  - **Anthropic Claude**: Claude Sonnet for vision tasks
  - **OpenAI GPT-4 Vision**: GPT-4o for vision tasks
- **VLM Management Page**: Configure providers, install models, test connections
- **Class Management**:
  - Add, edit, rename classes with custom colors
  - Delete classes and all associated annotations
  - Filter images by class (has/missing class)
  - Remap annotations between classes
  - Detect and manage orphaned class annotations

### Training & Virtual Environments
- **Virtual Environment Management**: Create isolated Python environments, clone GitHub repos, manage dependencies
- **Quick Setup**: One-click setup for axis_yolov5 and DetectX environments with correct Python versions and dependencies
- **Training Job Control**: Start, stop, pause, resume training with real-time progress and live log streaming
- **Training Presets**: Save and reuse configurations (Small/Medium/Large defaults included)
- **Training Queue**: Queue multiple jobs with priority-based execution and scheduling
- **YAML Configuration Editor**: Monaco Editor with syntax highlighting for dataset configs

### Model Export & Deployment
- **TFLite Export**: Convert trained models to TensorFlow Lite format with INT8 quantization
- **Axis YOLOv5 Workflow**: Specialized support for Axis camera-optimized YOLOv5 (ReLU6 activation)
- **DetectX ACAP Builder**: Generate ACAP packages (.eap) for deployment on Axis network cameras
  - ARTPEC-8 DLPU, ARTPEC-9 DLPU, Google Edge TPU support
  - CPU fallback option for debugging DLPU issues
  - Configurable detection thresholds (objectness, confidence, NMS)

### Monitoring & Tools
- **System Dashboard**: Real-time CPU, GPU, memory, and disk usage monitoring
- **System Logs**: View backend, frontend, training, and VLM logs with filtering
  - Clear logs by source (backend, frontend, training, VLM)
  - Real-time log tailing with auto-refresh
- **Web Terminal**: Interactive browser-based terminal with full PTY support
- **Support Page**: Donation and contribution links

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
ml-training-platform/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── database.py             # SQLAlchemy models
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Backend container (includes pyenv + Python 3.9)
│   ├── presets/                # Venv preset configurations
│   │   ├── axis_yolov5_requirements.txt  # TensorFlow 2.11, Python 3.9
│   │   ├── detectx_requirements.txt      # TensorFlow 2.20, Python 3.11+
│   │   └── florence2_requirements.txt    # Florence-2 VLM dependencies
│   └── api/
│       ├── annotations.py      # Annotation projects, images, labels
│       ├── autolabel.py        # Auto-labeling with pretrained models
│       ├── vlm_management.py   # VLM provider management (Ollama, cloud APIs)
│       ├── vlm_providers.py    # VLM provider implementations
│       ├── vlm_inference/      # Local VLM inference scripts
│       │   └── florence2_infer.py  # Florence-2 subprocess inference
│       ├── system_logs.py      # System log viewing and management
│       ├── datasets.py         # Dataset upload, browse, preview
│       ├── training.py         # Training job control
│       ├── venv.py             # Virtual environment management + presets
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
│   │       ├── VirtualEnvs.jsx     # Environment management + Quick Setup
│   │       ├── Presets.jsx         # Training presets
│   │       ├── Queue.jsx           # Training queue
│   │       ├── Exports.jsx         # Model exports
│   │       ├── VLM.jsx             # VLM management page
│   │       ├── SystemLogs.jsx      # System log viewer
│   │       ├── AxisYOLOv5.jsx      # Axis workflow
│   │       ├── DetectXBuild.jsx    # ACAP builder
│   │       ├── Terminal.jsx        # Web terminal
│   │       └── Support.jsx         # Donation page
│   └── dist/                   # Production build (generated)
│
├── docker-compose.yml          # Docker orchestration
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
- Python 3.11+ (for main app)
- Node.js 18+
- Git
- (Optional) NVIDIA GPU with CUDA
- (Optional) Docker for containerized deployment

### Option 1: Docker (Recommended)

The easiest way to get started:

```bash
git clone https://github.com/Prudencia/ml-training-platform.git
cd ml-training-platform
docker compose up --build -d
```

**Note:** GPU support is enabled by default. If you don't have an NVIDIA GPU or nvidia-container-toolkit installed, comment out the `deploy` section in `docker-compose.yml` before building (see [GPU Support](#gpu-support-in-docker) for details).

Access:
- **Frontend**: `http://localhost:3080`
- **Backend API**: `http://localhost:8081`
- **API Docs**: `http://localhost:8081/docs`

The Docker image includes pyenv with Python 3.9.19 pre-installed for axis_yolov5 compatibility.

### Option 2: Manual Setup

```bash
# Clone repository
git clone https://github.com/Prudencia/ml-training-platform.git
cd ml-training-platform

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup frontend
cd ../frontend
npm install
npm run build

# Start backend (serves both API and built frontend)
cd ../backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

Access everything at `http://localhost:8000`.

### Option 3: Development Mode

Run frontend and backend separately for development:

**Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm run dev
```

Access:
- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`

## Virtual Environment Setup

The platform requires two specialized virtual environments for the Axis workflow:

### axis_yolov5
- **Purpose**: YOLOv5 training with Axis-optimized export (ReLU6 activation)
- **Python**: 3.9.19 (required for TensorFlow 2.11)
- **Key packages**: TensorFlow 2.11.1, Keras 2.11.0, PyTorch, NumPy <2.0

### DetectX
- **Purpose**: Building ACAP packages (.eap) for Axis cameras
- **Python**: 3.11+ (system default)
- **Key packages**: TensorFlow 2.20.0, Keras 3.12.0

**Quick Setup**: Go to Virtual Environments page and click "Setup" for each preset. The platform will:
1. Install the required Python version via pyenv (if needed)
2. Create the virtual environment
3. Install all dependencies
4. Apply necessary patches (for axis_yolov5)

**Note**: For axis_yolov5, pyenv must be installed on the system. The Docker image has this pre-configured.

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
6. Setup Venv        → /venvs (use Quick Setup for axis_yolov5)
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
| `/api/vlm/` | VLM provider management, Ollama models, cloud API keys |
| `/api/training/` | Training job control, logs |
| `/api/venv/` | Virtual environment management, presets |
| `/api/presets/` | Training presets |
| `/api/workflows/` | Axis YOLOv5, export, DetectX |
| `/api/system/` | System monitoring |
| `/api/terminal/` | Web terminal |
| `/api/settings/` | User authentication, settings |
| `/api/logs/` | System log viewing and clearing |

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

## Docker Configuration

Default ports (can be changed in `docker-compose.yml`):
- **Frontend**: 3080
- **Backend**: 8081

To use different ports, edit `docker-compose.yml`:
```yaml
services:
  backend:
    ports:
      - "YOUR_PORT:8000"
  frontend:
    args:
      - VITE_API_URL=http://localhost:YOUR_PORT
    ports:
      - "YOUR_FRONTEND_PORT:80"
```

### Ollama (VLM Support)

The docker-compose includes an **Ollama service** for running local vision language models. This enables free, local auto-labeling with models like LLaVA, LLaMA 3.2 Vision, and more.

**Included by default:**
- Ollama runs as a Docker service with GPU support
- Backend connects to `http://ollama:11434` automatically
- Model data persisted in `ollama_data` volume

**To pull a vision model:**
1. Go to the VLM page in the web UI
2. Click "Pull" on any available model (e.g., LLaVA 7B)
3. Or use the custom model input to pull any Ollama model

**Cloud providers** (NVIDIA NIM, Anthropic, OpenAI) can also be configured in the VLM page with API keys.

### GPU Support in Docker

GPU support is **enabled by default** in `docker-compose.yml` using the deploy syntax. This requires nvidia-container-toolkit to be installed on the host system.

**If you don't have an NVIDIA GPU**, comment out the deploy section in `docker-compose.yml`:
```yaml
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
```

**If GPU is not detected** (works on WSL2 and Native Linux with nvidia-container-toolkit):
```bash
# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Rebuild container
docker compose down
docker compose up --build -d

# Verify GPU is accessible
docker exec -it ml-training-platform-backend-1 nvidia-smi
```

**Alternative: CDI mode**

If the default deploy syntax doesn't work, you can try CDI mode instead. Edit `docker-compose.yml`:
```yaml
    # Comment out the deploy section, then add:
    devices:
      - nvidia.com/gpu=all
```

CDI mode requires additional setup:
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
echo '{"features":{"cdi":true}}' | sudo tee /etc/docker/daemon.json
sudo service docker restart
```

**Rebuilding after changes:**
```bash
docker compose down
docker compose up --build -d

# Verify GPU is accessible inside the container
docker exec -it ml-training-platform-backend-1 nvidia-smi
```

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

**axis_yolov5 setup failing:**
- Ensure pyenv is installed
- Check Python 3.9.19 build dependencies are installed
- Docker image has this pre-configured

**Port conflicts:**
- Docker uses ports 3080/8081 to avoid conflicts with common services
- Change ports in `docker-compose.yml` if needed

## Support

If you find this project useful, consider supporting its development:
- Visit the Support page in the app
- Star the repository on GitHub
- Report bugs and suggest features
- Contribute code improvements

## License

**PolyForm Noncommercial License 1.0.0**

This software is free to use for:
- Personal projects and learning
- Research and education
- Non-profit organizations
- Hobby and amateur use

**Commercial use requires a separate license.** If you want to use this software for commercial purposes (selling products/services, internal business tools, etc.), please contact the author.

See the [LICENSE](LICENSE) file for the full legal text.

## Contributing

Contributions welcome! Please open an issue or PR.

---

**Built for the Axis camera ML deployment workflow**
