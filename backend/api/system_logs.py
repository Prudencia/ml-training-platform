"""
System Logs API - View and search various system logs for debugging
"""

from fastapi import APIRouter, Query, HTTPException
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import subprocess
import os
import re

router = APIRouter(prefix="/api/logs", tags=["logs"])

LOG_PATH = Path("storage/logs")
VENV_PATH = Path("storage/venvs")


@router.get("/sources")
async def get_log_sources():
    """List all available log sources"""
    sources = []

    # Docker backend logs
    sources.append({
        "id": "docker-backend",
        "name": "Backend Container",
        "description": "Docker container stdout/stderr logs",
        "type": "docker"
    })

    # Venv setup logs
    venv_logs = list(LOG_PATH.glob("venv_setup_*.log"))
    for log in sorted(venv_logs, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
        # Parse venv name from filename
        match = re.match(r'venv_setup_(.+?)_\d+\.log', log.name)
        venv_name = match.group(1) if match else log.stem
        sources.append({
            "id": f"venv:{log.name}",
            "name": f"Venv Setup: {venv_name}",
            "description": f"Setup log from {datetime.fromtimestamp(log.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}",
            "type": "file",
            "path": str(log)
        })

    # Training logs
    training_logs = list(LOG_PATH.glob("training_*.log"))
    for log in sorted(training_logs, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
        sources.append({
            "id": f"training:{log.name}",
            "name": f"Training: {log.stem}",
            "description": f"Training log from {datetime.fromtimestamp(log.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}",
            "type": "file",
            "path": str(log)
        })

    # VLM/Auto-label logs (from docker logs, filtered)
    sources.append({
        "id": "vlm",
        "name": "VLM Auto-Label",
        "description": "VLM inference and auto-labeling logs (filtered from backend)",
        "type": "docker-filter",
        "filter": "VLM"
    })

    return {"sources": sources}


@router.get("/docker")
async def get_docker_logs(
    lines: int = Query(200, ge=10, le=5000),
    filter: Optional[str] = None
):
    """Get Docker backend container logs"""
    try:
        result = subprocess.run(
            ["docker", "logs", "trainplattform-backend-1", "--tail", str(lines)],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Combine stdout and stderr
        logs = result.stdout + result.stderr

        # Apply filter if specified
        if filter:
            filter_lower = filter.lower()
            lines_list = logs.split('\n')
            lines_list = [l for l in lines_list if filter_lower in l.lower()]
            logs = '\n'.join(lines_list)

        return {
            "source": "docker-backend",
            "lines": logs.count('\n') + 1 if logs else 0,
            "content": logs
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timeout getting docker logs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/file/{filename:path}")
async def get_file_log(
    filename: str,
    lines: int = Query(500, ge=10, le=10000),
    filter: Optional[str] = None
):
    """Get a specific log file"""
    # Security: only allow logs from storage/logs
    log_path = LOG_PATH / filename
    if not log_path.exists():
        # Try full path if it starts with storage/logs
        log_path = Path(filename)
        if not str(log_path).startswith("storage/logs"):
            raise HTTPException(status_code=403, detail="Access denied")

    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # Get last N lines
        lines_list = content.split('\n')
        if len(lines_list) > lines:
            lines_list = lines_list[-lines:]

        # Apply filter if specified
        if filter:
            filter_lower = filter.lower()
            lines_list = [l for l in lines_list if filter_lower in l.lower()]

        content = '\n'.join(lines_list)

        return {
            "source": filename,
            "lines": len(lines_list),
            "content": content,
            "total_size": log_path.stat().st_size,
            "modified": datetime.fromtimestamp(log_path.stat().st_mtime).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vlm")
async def get_vlm_logs(
    lines: int = Query(500, ge=10, le=5000)
):
    """Get VLM/Auto-label related logs from docker"""
    try:
        result = subprocess.run(
            ["docker", "logs", "trainplattform-backend-1", "--tail", str(lines * 3)],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Combine and filter for VLM-related lines
        all_logs = result.stdout + result.stderr
        lines_list = all_logs.split('\n')

        vlm_keywords = ['vlm', 'auto-label', 'autolabel', 'florence', 'deepseek',
                       'inference', 'detection', 'bbox', 'provider']
        filtered = []
        for line in lines_list:
            line_lower = line.lower()
            if any(kw in line_lower for kw in vlm_keywords):
                filtered.append(line)

        # Limit to requested lines
        if len(filtered) > lines:
            filtered = filtered[-lines:]

        return {
            "source": "vlm",
            "lines": len(filtered),
            "content": '\n'.join(filtered)
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timeout getting logs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/venv/{venv_name}")
async def get_venv_logs(venv_name: str):
    """Get latest venv setup log for a specific venv"""
    # Find latest log file for this venv
    pattern = f"venv_setup_{venv_name}_*.log"
    logs = sorted(LOG_PATH.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)

    if not logs:
        raise HTTPException(status_code=404, detail=f"No setup logs found for venv: {venv_name}")

    latest_log = logs[0]

    try:
        with open(latest_log, 'r') as f:
            content = f.read()

        return {
            "source": f"venv:{venv_name}",
            "filename": latest_log.name,
            "lines": content.count('\n') + 1,
            "content": content,
            "modified": datetime.fromtimestamp(latest_log.stat().st_mtime).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors")
async def get_error_logs(
    lines: int = Query(200, ge=10, le=2000)
):
    """Get error-related logs from docker backend"""
    try:
        result = subprocess.run(
            ["docker", "logs", "trainplattform-backend-1", "--tail", str(lines * 5)],
            capture_output=True,
            text=True,
            timeout=10
        )

        all_logs = result.stdout + result.stderr
        lines_list = all_logs.split('\n')

        error_keywords = ['error', 'exception', 'traceback', 'failed', 'fatal', 'critical']
        filtered = []
        in_traceback = False

        for line in lines_list:
            line_lower = line.lower()

            # Start capturing on error keywords
            if any(kw in line_lower for kw in error_keywords):
                in_traceback = True
                filtered.append(line)
            # Continue capturing traceback lines
            elif in_traceback:
                if line.startswith(' ') or line.startswith('\t') or 'File "' in line:
                    filtered.append(line)
                else:
                    in_traceback = False

        if len(filtered) > lines:
            filtered = filtered[-lines:]

        return {
            "source": "errors",
            "lines": len(filtered),
            "content": '\n'.join(filtered)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_logs(
    query: str = Query(..., min_length=2),
    source: str = Query("docker", description="Log source: docker, vlm, errors"),
    lines: int = Query(500, ge=10, le=5000),
    case_sensitive: bool = Query(False)
):
    """Search across logs for a specific query"""
    try:
        # Get logs based on source
        if source == "docker":
            result = subprocess.run(
                ["docker", "logs", "trainplattform-backend-1", "--tail", str(lines * 3)],
                capture_output=True,
                text=True,
                timeout=10
            )
            all_logs = result.stdout + result.stderr
        elif source == "vlm":
            response = await get_vlm_logs(lines=lines * 2)
            all_logs = response["content"]
        elif source == "errors":
            response = await get_error_logs(lines=lines * 2)
            all_logs = response["content"]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown source: {source}")

        # Search
        lines_list = all_logs.split('\n')
        if case_sensitive:
            filtered = [l for l in lines_list if query in l]
        else:
            query_lower = query.lower()
            filtered = [l for l in lines_list if query_lower in l.lower()]

        if len(filtered) > lines:
            filtered = filtered[-lines:]

        return {
            "query": query,
            "source": source,
            "matches": len(filtered),
            "content": '\n'.join(filtered)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
