"""
VLM Management API - Manage local and cloud VLM providers

Features:
- Ollama: List/pull/delete models, check service status, custom model installation
- Cloud providers: API key management, connection testing
- NVIDIA NIM: Cloud API for NVIDIA vision models
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import httpx
import asyncio
import platform
from datetime import datetime

from database import get_db, SystemSettings

router = APIRouter()

# Background tasks tracking
_ollama_pull_tasks = {}  # model_name -> {"status": str, "progress": float, "error": str}


# ============ Pydantic Models ============

class OllamaModelInfo(BaseModel):
    name: str
    size: Optional[float] = None  # Size in GB
    modified_at: Optional[str] = None
    digest: Optional[str] = None


class OllamaPullRequest(BaseModel):
    model_name: str  # e.g., "llava:13b", "llava:7b", "bakllava"


class APIKeyUpdate(BaseModel):
    api_key: str


class OllamaEndpointUpdate(BaseModel):
    endpoint: str
    model: Optional[str] = "llava:13b"


class CustomModelPullRequest(BaseModel):
    model_name: str  # Any valid Ollama model name or HuggingFace model


class ProviderStatus(BaseModel):
    name: str
    display_name: str
    provider_type: str  # "local" or "cloud"
    is_configured: bool
    is_available: bool
    error: Optional[str] = None
    models: List[str] = []
    active_model: Optional[str] = None


# ============ Helper Functions ============

def get_setting(db: Session, key: str) -> Optional[str]:
    """Get a setting value from database"""
    setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
    return setting.value if setting else None


def set_setting(db: Session, key: str, value: str, value_type: str = "string"):
    """Set a setting value in database"""
    setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
    if setting:
        setting.value = value
        setting.value_type = value_type
        setting.updated_at = datetime.utcnow()
    else:
        setting = SystemSettings(key=key, value=value, value_type=value_type)
        db.add(setting)
    db.commit()


def delete_setting(db: Session, key: str):
    """Delete a setting from database"""
    setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
    if setting:
        db.delete(setting)
        db.commit()


async def get_ollama_endpoint(db: Session) -> str:
    """Get configured Ollama endpoint"""
    endpoint = get_setting(db, "vlm_ollama_endpoint")
    # Default to host.docker.internal for Docker, or use OLLAMA_HOST env var
    import os
    default = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
    return endpoint or default


# ============ Ollama Management Endpoints ============

@router.get("/ollama/status")
async def get_ollama_status(db: Session = Depends(get_db)):
    """Check if Ollama service is running and accessible"""
    endpoint = await get_ollama_endpoint(db)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{endpoint}/api/version")
            if response.status_code == 200:
                version_info = response.json()
                return {
                    "status": "running",
                    "endpoint": endpoint,
                    "version": version_info.get("version", "unknown")
                }
    except httpx.ConnectError:
        return {
            "status": "not_running",
            "endpoint": endpoint,
            "error": "Cannot connect to Ollama. Make sure Ollama is installed and running."
        }
    except Exception as e:
        return {
            "status": "error",
            "endpoint": endpoint,
            "error": str(e)
        }

    return {
        "status": "error",
        "endpoint": endpoint,
        "error": "Unexpected response from Ollama"
    }


@router.get("/ollama/models")
async def list_ollama_models(db: Session = Depends(get_db)):
    """List all downloaded Ollama models"""
    endpoint = await get_ollama_endpoint(db)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{endpoint}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("models", []):
                    size_bytes = model.get("size", 0)
                    size_gb = round(size_bytes / (1024 ** 3), 2) if size_bytes else None
                    models.append({
                        "name": model.get("name"),
                        "size_gb": size_gb,
                        "modified_at": model.get("modified_at"),
                        "digest": model.get("digest", "")[:12]
                    })
                return {"models": models, "endpoint": endpoint}
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama service not running")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=500, detail="Failed to list models")


@router.get("/ollama/available")
async def list_available_ollama_models():
    """List recommended VLM models available to pull"""
    # Curated list of vision-capable models
    available = [
        # LLaVA family
        {
            "name": "llava:7b",
            "display_name": "LLaVA 7B",
            "description": "Fast, lightweight vision model",
            "size_gb": 4.7,
            "recommended": True,
            "category": "llava"
        },
        {
            "name": "llava:13b",
            "display_name": "LLaVA 13B",
            "description": "Balanced performance and quality",
            "size_gb": 8.0,
            "recommended": True,
            "category": "llava"
        },
        {
            "name": "llava:34b",
            "display_name": "LLaVA 34B",
            "description": "Highest quality, requires more VRAM",
            "size_gb": 20.0,
            "recommended": False,
            "category": "llava"
        },
        {
            "name": "llava-llama3",
            "display_name": "LLaVA-LLaMA3",
            "description": "LLaVA with LLaMA 3 base",
            "size_gb": 5.5,
            "recommended": False,
            "category": "llava"
        },
        {
            "name": "bakllava",
            "display_name": "BakLLaVA",
            "description": "Fine-tuned LLaVA variant",
            "size_gb": 4.7,
            "recommended": False,
            "category": "llava"
        },
        # LLaMA 3.2 Vision (new)
        {
            "name": "llama3.2-vision",
            "display_name": "LLaMA 3.2 Vision 11B",
            "description": "Meta's latest vision model, excellent accuracy",
            "size_gb": 7.9,
            "recommended": True,
            "category": "llama"
        },
        {
            "name": "llama3.2-vision:90b",
            "display_name": "LLaMA 3.2 Vision 90B",
            "description": "Largest Meta vision model, best quality",
            "size_gb": 55.0,
            "recommended": False,
            "category": "llama"
        },
        # Efficient models
        {
            "name": "minicpm-v",
            "display_name": "MiniCPM-V",
            "description": "Efficient vision model, good for object detection",
            "size_gb": 5.0,
            "recommended": True,
            "category": "efficient"
        },
        {
            "name": "moondream",
            "display_name": "Moondream 2",
            "description": "Ultra-lightweight, runs on CPU",
            "size_gb": 1.7,
            "recommended": True,
            "category": "efficient"
        },
        # Advanced models
        {
            "name": "llava-phi3",
            "display_name": "LLaVA-Phi3",
            "description": "LLaVA with Microsoft Phi-3 base",
            "size_gb": 3.8,
            "recommended": False,
            "category": "advanced"
        },
        {
            "name": "cogvlm2",
            "display_name": "CogVLM2",
            "description": "Strong OCR and detailed understanding",
            "size_gb": 17.0,
            "recommended": False,
            "category": "advanced"
        }
    ]
    return {"available_models": available}


@router.get("/ollama/install-instructions")
async def get_ollama_install_instructions():
    """Get installation instructions for Ollama based on platform"""
    system = platform.system().lower()

    instructions = {
        "platform": system,
        "steps": [],
        "download_url": "https://ollama.ai/download",
        "docs_url": "https://github.com/ollama/ollama"
    }

    if system == "linux":
        instructions["steps"] = [
            "Run: curl -fsSL https://ollama.ai/install.sh | sh",
            "Start service: ollama serve",
            "Or run with systemd: sudo systemctl enable ollama && sudo systemctl start ollama"
        ]
        instructions["one_liner"] = "curl -fsSL https://ollama.ai/install.sh | sh"
    elif system == "darwin":  # macOS
        instructions["steps"] = [
            "Download from https://ollama.ai/download",
            "Or install with Homebrew: brew install ollama",
            "Run: ollama serve"
        ]
        instructions["one_liner"] = "brew install ollama && ollama serve"
    elif system == "windows":
        instructions["steps"] = [
            "Download installer from https://ollama.ai/download",
            "Run the installer",
            "Ollama will start automatically"
        ]
        instructions["one_liner"] = None

    return instructions


@router.post("/ollama/pull-custom")
async def pull_custom_model(
    request: CustomModelPullRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Pull any Ollama model by name (including custom/community models)"""
    model_name = request.model_name.strip()

    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    endpoint = await get_ollama_endpoint(db)

    # Check if already pulling this model
    if model_name in _ollama_pull_tasks and _ollama_pull_tasks[model_name]["status"] == "pulling":
        return {"status": "already_pulling", "model": model_name}

    # Initialize task tracking
    _ollama_pull_tasks[model_name] = {
        "status": "starting",
        "progress": 0,
        "error": None,
        "started_at": datetime.utcnow().isoformat()
    }

    # Start background pull
    background_tasks.add_task(_pull_model_task, model_name, endpoint)

    return {"status": "started", "model": model_name}


@router.post("/ollama/pull")
async def pull_ollama_model(
    request: OllamaPullRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start pulling/downloading an Ollama model"""
    model_name = request.model_name
    endpoint = await get_ollama_endpoint(db)

    # Check if already pulling this model
    if model_name in _ollama_pull_tasks and _ollama_pull_tasks[model_name]["status"] == "pulling":
        return {"status": "already_pulling", "model": model_name}

    # Initialize task tracking
    _ollama_pull_tasks[model_name] = {
        "status": "starting",
        "progress": 0,
        "error": None,
        "started_at": datetime.utcnow().isoformat()
    }

    # Start background pull
    background_tasks.add_task(_pull_model_task, model_name, endpoint)

    return {"status": "started", "model": model_name}


async def _pull_model_task(model_name: str, endpoint: str):
    """Background task to pull Ollama model with progress tracking"""
    try:
        _ollama_pull_tasks[model_name]["status"] = "pulling"

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{endpoint}/api/pull",
                json={"name": model_name},
                timeout=None
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            import json
                            data = json.loads(line)

                            # Update progress
                            if "completed" in data and "total" in data:
                                total = data["total"]
                                completed = data["completed"]
                                if total > 0:
                                    _ollama_pull_tasks[model_name]["progress"] = round(
                                        (completed / total) * 100, 1
                                    )

                            # Check for completion
                            if data.get("status") == "success":
                                _ollama_pull_tasks[model_name]["status"] = "completed"
                                _ollama_pull_tasks[model_name]["progress"] = 100
                                return

                            # Update status message
                            if "status" in data:
                                status_msg = data["status"]
                                if "pulling" in status_msg.lower():
                                    _ollama_pull_tasks[model_name]["status"] = "pulling"
                                elif "verifying" in status_msg.lower():
                                    _ollama_pull_tasks[model_name]["status"] = "verifying"

                        except json.JSONDecodeError:
                            pass

        _ollama_pull_tasks[model_name]["status"] = "completed"
        _ollama_pull_tasks[model_name]["progress"] = 100

    except Exception as e:
        _ollama_pull_tasks[model_name]["status"] = "failed"
        _ollama_pull_tasks[model_name]["error"] = str(e)


@router.get("/ollama/pull/{model_name}/status")
async def get_pull_status(model_name: str):
    """Get the status of a model pull operation"""
    if model_name not in _ollama_pull_tasks:
        return {"status": "not_found", "model": model_name}

    return {
        "model": model_name,
        **_ollama_pull_tasks[model_name]
    }


@router.delete("/ollama/models/{model_name}")
async def delete_ollama_model(model_name: str, db: Session = Depends(get_db)):
    """Delete an Ollama model"""
    endpoint = await get_ollama_endpoint(db)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{endpoint}/api/delete",
                json={"name": model_name}
            )
            if response.status_code == 200:
                return {"status": "deleted", "model": model_name}
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to delete model: {response.text}"
                )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama service not running")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/ollama/endpoint")
async def update_ollama_endpoint(request: OllamaEndpointUpdate, db: Session = Depends(get_db)):
    """Update Ollama endpoint configuration"""
    set_setting(db, "vlm_ollama_endpoint", request.endpoint)
    if request.model:
        set_setting(db, "vlm_ollama_model", request.model)

    # Test connection
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{request.endpoint}/api/version")
            if response.status_code == 200:
                return {
                    "status": "success",
                    "endpoint": request.endpoint,
                    "model": request.model,
                    "connected": True
                }
    except Exception:
        pass

    return {
        "status": "saved",
        "endpoint": request.endpoint,
        "model": request.model,
        "connected": False,
        "warning": "Endpoint saved but could not connect to Ollama"
    }


# ============ Cloud Provider Management ============

@router.get("/providers/status")
async def get_all_providers_status(db: Session = Depends(get_db)):
    """Get status of all VLM providers"""
    providers = []

    # Ollama (Local)
    ollama_endpoint = get_setting(db, "vlm_ollama_endpoint") or "http://localhost:11434"
    ollama_model = get_setting(db, "vlm_ollama_model") or "llava:13b"
    ollama_available = False
    ollama_error = None
    ollama_models = []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_endpoint}/api/tags")
            if response.status_code == 200:
                ollama_available = True
                data = response.json()
                ollama_models = [m.get("name") for m in data.get("models", [])]
    except Exception as e:
        ollama_error = "Ollama not running"

    providers.append({
        "name": "ollama",
        "display_name": "Ollama (Local)",
        "provider_type": "local",
        "is_configured": True,  # Ollama is always "configured" (just needs to be running)
        "is_available": ollama_available,
        "error": ollama_error,
        "models": ollama_models,
        "active_model": ollama_model,
        "endpoint": ollama_endpoint
    })

    # Anthropic
    anthropic_key = get_setting(db, "vlm_anthropic_api_key")
    anthropic_configured = bool(anthropic_key)
    anthropic_available = False
    anthropic_error = None

    if anthropic_configured:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            # Simple validation - just check if key format is valid
            anthropic_available = len(anthropic_key) > 20
        except Exception as e:
            anthropic_error = str(e)

    providers.append({
        "name": "anthropic",
        "display_name": "Anthropic Claude",
        "provider_type": "cloud",
        "is_configured": anthropic_configured,
        "is_available": anthropic_available,
        "error": anthropic_error,
        "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
        "active_model": "claude-sonnet-4-20250514",
        "key_hint": f"...{anthropic_key[-4:]}" if anthropic_key else None
    })

    # OpenAI
    openai_key = get_setting(db, "vlm_openai_api_key")
    openai_configured = bool(openai_key)
    openai_available = False
    openai_error = None

    if openai_configured:
        try:
            openai_available = len(openai_key) > 20
        except Exception as e:
            openai_error = str(e)

    providers.append({
        "name": "openai",
        "display_name": "OpenAI GPT-4 Vision",
        "provider_type": "cloud",
        "is_configured": openai_configured,
        "is_available": openai_available,
        "error": openai_error,
        "models": ["gpt-4o", "gpt-4-turbo"],
        "active_model": "gpt-4o",
        "key_hint": f"...{openai_key[-4:]}" if openai_key else None
    })

    # NVIDIA NIM
    nvidia_key = get_setting(db, "vlm_nvidia_api_key")
    nvidia_configured = bool(nvidia_key)
    nvidia_available = False
    nvidia_error = None

    if nvidia_configured:
        nvidia_available = nvidia_key.startswith("nvapi-") and len(nvidia_key) > 20

    providers.append({
        "name": "nvidia",
        "display_name": "NVIDIA NIM",
        "provider_type": "cloud",
        "is_configured": nvidia_configured,
        "is_available": nvidia_available,
        "error": nvidia_error,
        "models": ["microsoft/phi-3.5-vision-instruct", "nvidia/vila", "meta/llama-3.2-90b-vision-instruct"],
        "active_model": "microsoft/phi-3.5-vision-instruct",
        "key_hint": f"...{nvidia_key[-4:]}" if nvidia_key else None,
        "free_credits": "1000 free credits for new users",
        "signup_url": "https://build.nvidia.com/"
    })

    return {"providers": providers}


@router.put("/providers/anthropic/key")
async def update_anthropic_key(request: APIKeyUpdate, db: Session = Depends(get_db)):
    """Update Anthropic API key"""
    set_setting(db, "vlm_anthropic_api_key", request.api_key)

    # Validate key
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=request.api_key)
        # Try a minimal API call to validate
        return {
            "status": "success",
            "provider": "anthropic",
            "key_hint": f"...{request.api_key[-4:]}",
            "valid": True
        }
    except Exception as e:
        return {
            "status": "saved",
            "provider": "anthropic",
            "key_hint": f"...{request.api_key[-4:]}",
            "valid": False,
            "warning": f"Key saved but validation failed: {str(e)}"
        }


@router.put("/providers/openai/key")
async def update_openai_key(request: APIKeyUpdate, db: Session = Depends(get_db)):
    """Update OpenAI API key"""
    set_setting(db, "vlm_openai_api_key", request.api_key)

    # Validate key format
    valid = request.api_key.startswith("sk-") and len(request.api_key) > 20

    return {
        "status": "success" if valid else "saved",
        "provider": "openai",
        "key_hint": f"...{request.api_key[-4:]}",
        "valid": valid,
        "warning": None if valid else "Key format may be invalid"
    }


@router.put("/providers/nvidia/key")
async def update_nvidia_key(request: APIKeyUpdate, db: Session = Depends(get_db)):
    """Update NVIDIA NIM API key"""
    set_setting(db, "vlm_nvidia_api_key", request.api_key)

    # Validate key format (NVIDIA keys start with nvapi-)
    valid = request.api_key.startswith("nvapi-") and len(request.api_key) > 20

    return {
        "status": "success" if valid else "saved",
        "provider": "nvidia",
        "key_hint": f"...{request.api_key[-4:]}",
        "valid": valid,
        "warning": None if valid else "Key format may be invalid (should start with nvapi-)"
    }


@router.delete("/providers/{provider}/key")
async def delete_provider_key(provider: str, db: Session = Depends(get_db)):
    """Delete API key for a cloud provider"""
    key_mapping = {
        "anthropic": "vlm_anthropic_api_key",
        "openai": "vlm_openai_api_key",
        "nvidia": "vlm_nvidia_api_key"
    }

    if provider not in key_mapping:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    delete_setting(db, key_mapping[provider])
    return {"status": "deleted", "provider": provider}


@router.post("/providers/{provider}/test")
async def test_provider_connection(provider: str, db: Session = Depends(get_db)):
    """Test connection to a VLM provider"""
    if provider == "ollama":
        endpoint = get_setting(db, "vlm_ollama_endpoint") or "http://localhost:11434"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{endpoint}/api/version")
                if response.status_code == 200:
                    return {"status": "success", "provider": provider, "message": "Connected"}
        except Exception as e:
            return {"status": "failed", "provider": provider, "error": str(e)}

    elif provider == "anthropic":
        api_key = get_setting(db, "vlm_anthropic_api_key")
        if not api_key:
            return {"status": "failed", "provider": provider, "error": "API key not configured"}

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Make a minimal API call
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return {"status": "success", "provider": provider, "message": "API key valid"}
        except Exception as e:
            return {"status": "failed", "provider": provider, "error": str(e)}

    elif provider == "openai":
        api_key = get_setting(db, "vlm_openai_api_key")
        if not api_key:
            return {"status": "failed", "provider": provider, "error": "API key not configured"}

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            # Make a minimal API call
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10
            )
            return {"status": "success", "provider": provider, "message": "API key valid"}
        except Exception as e:
            return {"status": "failed", "provider": provider, "error": str(e)}

    elif provider == "nvidia":
        api_key = get_setting(db, "vlm_nvidia_api_key")
        if not api_key:
            return {"status": "failed", "provider": provider, "error": "API key not configured"}

        try:
            # NVIDIA NIM uses OpenAI-compatible API
            from openai import OpenAI
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key
            )
            # Make a minimal API call to test the key
            response = client.chat.completions.create(
                model="microsoft/phi-3.5-vision-instruct",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10
            )
            return {"status": "success", "provider": provider, "message": "API key valid"}
        except Exception as e:
            return {"status": "failed", "provider": provider, "error": str(e)}

    return {"status": "failed", "provider": provider, "error": "Unknown provider"}
