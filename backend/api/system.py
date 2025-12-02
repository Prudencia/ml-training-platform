from fastapi import APIRouter
import psutil
import platform
from pathlib import Path

router = APIRouter()

def get_cpu_model():
    """Get CPU model name from /proc/cpuinfo or platform"""
    # Try platform.processor() first
    proc = platform.processor()
    if proc:
        return proc

    # On Linux, read from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":")[1].strip()
    except:
        pass

    return "Unknown"


@router.get("/info")
async def get_system_info():
    """Get system information"""
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": get_cpu_model(),
        "python_version": platform.python_version()
    }

@router.get("/resources")
async def get_system_resources():
    """Get system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # Try to get GPU info (if nvidia-smi available)
    gpu_info = None
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in gpu_lines:
                parts = line.split(', ')
                if len(parts) == 4:
                    gpu_info.append({
                        "name": parts[0],
                        "memory_total_mb": int(parts[1]),
                        "memory_used_mb": int(parts[2]),
                        "utilization_percent": int(parts[3])
                    })
    except Exception:
        pass

    return {
        "cpu": {
            "percent": cpu_percent,
            "cores": psutil.cpu_count()
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "percent": disk.percent
        },
        "gpu": gpu_info
    }

@router.get("/storage")
async def get_storage_info():
    """Get storage information for datasets, models, etc."""
    def get_dir_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

    datasets_path = Path("storage/datasets")
    models_path = Path("storage/models")
    venvs_path = Path("storage/venvs")
    logs_path = Path("storage/logs")

    return {
        "datasets_gb": round(get_dir_size(datasets_path) / (1024**3), 2),
        "models_gb": round(get_dir_size(models_path) / (1024**3), 2),
        "venvs_gb": round(get_dir_size(venvs_path) / (1024**3), 2),
        "logs_gb": round(get_dir_size(logs_path) / (1024**3), 2)
    }
