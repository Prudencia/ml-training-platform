from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from pathlib import Path
import os

from api import (
    datasets,
    training,
    venv,
    presets,
    yaml_config,
    system,
    workflows,
    terminal,
    annotations,
    autolabel,
    settings
)
from database import init_db

app = FastAPI(title="ML Training Platform", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# Include API routers
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(venv.router, prefix="/api/venv", tags=["virtual-environments"])
app.include_router(presets.router, prefix="/api/presets", tags=["presets"])
app.include_router(yaml_config.router, prefix="/api/yaml", tags=["yaml-config"])
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(workflows.router, prefix="/api/workflows", tags=["workflows"])
app.include_router(terminal.router, prefix="/api/terminal", tags=["terminal"])
app.include_router(annotations.router, prefix="/api/annotations", tags=["annotations"])
app.include_router(autolabel.router, prefix="/api/autolabel", tags=["autolabel"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])

# Serve static files from frontend build
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

# Catch-all route to serve index.html for SPA routing
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # If path starts with /api, return 404
    if full_path.startswith("api/"):
        return {"message": "ML Training Platform API", "version": "1.0.0"}

    # Check if file exists in dist (for static files like shield.png)
    file_path = frontend_dist / full_path
    if file_path.is_file():
        return FileResponse(file_path)

    # Serve index.html for all other routes (SPA)
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return {"message": "Frontend build not found. Run 'npm run build' in frontend directory."}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
