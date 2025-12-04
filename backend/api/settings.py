from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import hashlib
import secrets
import json

from database import get_db, SystemSettings, User

router = APIRouter()

# Simple token storage (in production, use Redis or similar)
active_tokens = {}

# OAuth2 scheme (optional, only used when auth is enabled)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/settings/login", auto_error=False)


# ============ Pydantic Models ============

class SettingUpdate(BaseModel):
    value: str
    value_type: Optional[str] = "string"


class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    role: str = "user"


class UserUpdate(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict


# ============ Helper Functions ============

def hash_password(password: str) -> str:
    """Simple password hashing with salt"""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        salt, stored_hash = hashed.split(":")
        check_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return check_hash == stored_hash
    except:
        return False


def create_token(user_id: int) -> str:
    """Create a simple access token"""
    token = secrets.token_urlsafe(32)
    active_tokens[token] = {
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=24)
    }
    return token


def get_setting(key: str, db: Session, default=None):
    """Get a system setting value"""
    setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
    if not setting:
        return default

    if setting.value_type == "bool":
        return setting.value.lower() in ("true", "1", "yes")
    elif setting.value_type == "int":
        return int(setting.value)
    elif setting.value_type == "json":
        return json.loads(setting.value)
    return setting.value


def is_auth_enabled(db: Session) -> bool:
    """Check if authentication is enabled"""
    return get_setting("auth_enabled", db, default=False)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user from token (returns None if auth disabled or no token)"""
    # If auth is disabled, return None (allow all)
    if not is_auth_enabled(db):
        return None

    # If no token provided
    if not token:
        return None

    # Check token validity
    token_data = active_tokens.get(token)
    if not token_data:
        return None

    if datetime.utcnow() > token_data["expires_at"]:
        del active_tokens[token]
        return None

    return db.query(User).filter(User.id == token_data["user_id"]).first()


def require_auth(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Dependency that requires authentication when enabled"""
    if is_auth_enabled(db) and not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Dependency that requires admin role when auth is enabled"""
    if is_auth_enabled(db):
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
        if user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required",
            )
    return user


# ============ Settings Endpoints ============

@router.get("/")
async def get_all_settings(db: Session = Depends(get_db)):
    """Get all system settings"""
    settings = db.query(SystemSettings).all()

    # Initialize default settings if empty
    if not settings:
        defaults = [
            {"key": "auth_enabled", "value": "false", "value_type": "bool", "description": "Enable user authentication"},
            {"key": "allow_registration", "value": "false", "value_type": "bool", "description": "Allow new user registration"},
            {"key": "max_upload_size_mb", "value": "500", "value_type": "int", "description": "Maximum upload size in MB"},
        ]
        for d in defaults:
            setting = SystemSettings(**d)
            db.add(setting)
        db.commit()
        settings = db.query(SystemSettings).all()

    return [{
        "key": s.key,
        "value": s.value,
        "value_type": s.value_type,
        "description": s.description,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    } for s in settings]


@router.get("/{key}")
async def get_setting_value(key: str, db: Session = Depends(get_db)):
    """Get a specific setting"""
    setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
    if not setting:
        raise HTTPException(status_code=404, detail="Setting not found")

    return {
        "key": setting.key,
        "value": setting.value,
        "value_type": setting.value_type,
        "description": setting.description,
    }


@router.put("/{key}")
async def update_setting(
    key: str,
    update: SettingUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin)
):
    """Update a system setting (admin only when auth enabled)"""
    setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()

    if setting:
        setting.value = update.value
        if update.value_type:
            setting.value_type = update.value_type
        setting.updated_at = datetime.utcnow()
    else:
        setting = SystemSettings(
            key=key,
            value=update.value,
            value_type=update.value_type or "string",
        )
        db.add(setting)

    db.commit()
    return {"message": "Setting updated"}


@router.get("/auth/status")
async def get_auth_status(db: Session = Depends(get_db)):
    """Check if authentication is enabled"""
    auth_enabled = is_auth_enabled(db)
    allow_registration = get_setting("allow_registration", db, default=False)

    return {
        "auth_enabled": auth_enabled,
        "allow_registration": allow_registration,
    }


# ============ Authentication Endpoints ============

class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """Login and get access token"""
    if not is_auth_enabled(db):
        raise HTTPException(status_code=400, detail="Authentication is not enabled")

    user = db.query(User).filter(User.username == login_data.username).first()
    if not user or not verify_password(login_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not user.is_active:
        raise HTTPException(status_code=400, detail="User account is disabled")

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    token = create_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
        }
    }


@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    """Logout and invalidate token"""
    if token and token in active_tokens:
        del active_tokens[token]
    return {"message": "Logged out"}


@router.get("/me")
async def get_current_user_info(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get current user info"""
    if not is_auth_enabled(db):
        return {"auth_enabled": False}

    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None,
    }


# ============ User Management Endpoints ============

@router.get("/users")
async def list_users(
    db: Session = Depends(get_db),
    user: User = Depends(require_admin)
):
    """List all users (admin only)"""
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [{
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "role": u.role,
        "is_active": u.is_active,
        "created_at": u.created_at.isoformat() if u.created_at else None,
        "last_login": u.last_login.isoformat() if u.last_login else None,
    } for u in users]


@router.post("/users")
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Create a new user (admin only, or first user)"""
    # Check if this is the first user (no admin check needed)
    user_count = db.query(User).count()
    if user_count > 0 and not current_user:
        if is_auth_enabled(db):
            raise HTTPException(status_code=401, detail="Admin access required")

    # Check if username exists
    existing = db.query(User).filter(User.username == user_data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Create user
    new_user = User(
        username=user_data.username,
        password_hash=hash_password(user_data.password),
        email=user_data.email,
        role=user_data.role if user_count > 0 else "admin",  # First user is admin
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "id": new_user.id,
        "username": new_user.username,
        "role": new_user.role,
        "message": "User created"
    }


@router.post("/users/register")
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Self-registration (only if enabled)"""
    if not is_auth_enabled(db):
        raise HTTPException(status_code=400, detail="Authentication is not enabled")

    if not get_setting("allow_registration", db, default=False):
        raise HTTPException(status_code=400, detail="Registration is not allowed")

    # Check if username exists
    existing = db.query(User).filter(User.username == user_data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Create user (always as regular user)
    new_user = User(
        username=user_data.username,
        password_hash=hash_password(user_data.password),
        email=user_data.email,
        role="user",  # Self-registered users are always regular users
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "id": new_user.id,
        "username": new_user.username,
        "message": "Registration successful"
    }


@router.put("/users/{user_id}")
async def update_user(
    user_id: int,
    update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Update user (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if update.email is not None:
        user.email = update.email
    if update.role is not None:
        user.role = update.role
    if update.is_active is not None:
        user.is_active = update.is_active

    db.commit()
    return {"message": "User updated"}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Delete user (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent deleting yourself
    if current_user and user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    db.delete(user)
    db.commit()
    return {"message": "User deleted"}


@router.post("/users/{user_id}/change-password")
async def change_password(
    user_id: int,
    passwords: PasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Change user password"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Only allow users to change their own password, or admin to change any
    if current_user and current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Cannot change other user's password")

    # Verify current password
    if not verify_password(passwords.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    user.password_hash = hash_password(passwords.new_password)
    db.commit()
    return {"message": "Password changed"}


# ============ Setup Endpoint ============

@router.post("/setup/init")
async def initial_setup(
    admin_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Initial setup - create first admin user"""
    # Check if any users exist
    user_count = db.query(User).count()
    if user_count > 0:
        raise HTTPException(status_code=400, detail="Setup already completed")

    # Create admin user
    admin = User(
        username=admin_data.username,
        password_hash=hash_password(admin_data.password),
        email=admin_data.email,
        role="admin",
    )
    db.add(admin)

    # Enable auth by default after setup
    auth_setting = db.query(SystemSettings).filter(SystemSettings.key == "auth_enabled").first()
    if auth_setting:
        auth_setting.value = "true"
    else:
        db.add(SystemSettings(key="auth_enabled", value="true", value_type="bool"))

    db.commit()
    db.refresh(admin)

    return {
        "message": "Setup complete",
        "admin_id": admin.id,
        "username": admin.username,
    }


@router.get("/setup/status")
async def setup_status(db: Session = Depends(get_db)):
    """Check if initial setup is needed"""
    user_count = db.query(User).count()
    return {
        "setup_required": user_count == 0,
        "auth_enabled": is_auth_enabled(db),
    }


# ============ VLM Settings Endpoints ============

@router.get("/vlm")
async def get_vlm_settings(db: Session = Depends(get_db)):
    """Get VLM-related settings (API keys are masked)"""
    anthropic_key = get_setting("vlm_anthropic_api_key", db)
    openai_key = get_setting("vlm_openai_api_key", db)
    ollama_endpoint = get_setting("vlm_ollama_endpoint", db, default="http://localhost:11434")
    ollama_model = get_setting("vlm_ollama_model", db, default="llava:13b")

    return {
        "anthropic": {
            "is_configured": bool(anthropic_key),
            "key_preview": f"...{anthropic_key[-8:]}" if anthropic_key and len(anthropic_key) > 8 else None,
        },
        "openai": {
            "is_configured": bool(openai_key),
            "key_preview": f"...{openai_key[-8:]}" if openai_key and len(openai_key) > 8 else None,
        },
        "ollama": {
            "endpoint": ollama_endpoint,
            "model": ollama_model,
        }
    }


class VLMKeyUpdate(BaseModel):
    api_key: str


class VLMOllamaUpdate(BaseModel):
    endpoint: Optional[str] = None
    model: Optional[str] = None


@router.put("/vlm/anthropic")
async def update_anthropic_key(
    update: VLMKeyUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin)
):
    """Update Anthropic API key (admin only when auth enabled)"""
    key = "vlm_anthropic_api_key"
    setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()

    if setting:
        setting.value = update.api_key
        setting.updated_at = datetime.utcnow()
    else:
        setting = SystemSettings(
            key=key,
            value=update.api_key,
            value_type="string",
            description="Anthropic API key for VLM auto-labeling"
        )
        db.add(setting)

    db.commit()
    return {"message": "Anthropic API key updated"}


@router.put("/vlm/openai")
async def update_openai_key(
    update: VLMKeyUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin)
):
    """Update OpenAI API key (admin only when auth enabled)"""
    key = "vlm_openai_api_key"
    setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()

    if setting:
        setting.value = update.api_key
        setting.updated_at = datetime.utcnow()
    else:
        setting = SystemSettings(
            key=key,
            value=update.api_key,
            value_type="string",
            description="OpenAI API key for VLM auto-labeling"
        )
        db.add(setting)

    db.commit()
    return {"message": "OpenAI API key updated"}


@router.put("/vlm/ollama")
async def update_ollama_settings(
    update: VLMOllamaUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin)
):
    """Update Ollama settings (admin only when auth enabled)"""
    if update.endpoint is not None:
        key = "vlm_ollama_endpoint"
        setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
        if setting:
            setting.value = update.endpoint
            setting.updated_at = datetime.utcnow()
        else:
            setting = SystemSettings(
                key=key,
                value=update.endpoint,
                value_type="string",
                description="Ollama API endpoint for VLM auto-labeling"
            )
            db.add(setting)

    if update.model is not None:
        key = "vlm_ollama_model"
        setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
        if setting:
            setting.value = update.model
            setting.updated_at = datetime.utcnow()
        else:
            setting = SystemSettings(
                key=key,
                value=update.model,
                value_type="string",
                description="Default Ollama model for VLM auto-labeling"
            )
            db.add(setting)

    db.commit()
    return {"message": "Ollama settings updated"}


@router.delete("/vlm/{provider}")
async def delete_vlm_key(
    provider: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin)
):
    """Delete VLM provider API key (admin only when auth enabled)"""
    key_map = {
        "anthropic": "vlm_anthropic_api_key",
        "openai": "vlm_openai_api_key",
    }

    if provider not in key_map:
        raise HTTPException(status_code=400, detail="Invalid provider. Use 'anthropic' or 'openai'")

    key = key_map[provider]
    setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()

    if setting:
        db.delete(setting)
        db.commit()
        return {"message": f"{provider} API key deleted"}

    raise HTTPException(status_code=404, detail="API key not found")
