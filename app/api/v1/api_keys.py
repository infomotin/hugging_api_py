from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user
from models import User, ApiKey, AuditLog

router = APIRouter(prefix="/api-keys", tags=["API Keys Management"])

class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Friendly label for the API key")

@router.get("")
def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all API keys belonging to the current authenticated user."""
    keys = db.query(ApiKey).filter(ApiKey.user_id == current_user.id).order_by(ApiKey.created_at.desc()).all()
    return {
        "success": True,
        "keys": [k.to_dict() for k in keys]
    }

@router.post("", status_code=status.HTTP_201_CREATED)
def create_api_key(
    payload: CreateApiKeyRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate a new external API key (format: sk-ent-...).
    Note: The full secret token is ONLY returned in this response.
    """
    raw_token, key_obj = ApiKey.generate_key_pair(
        name=payload.name.strip(),
        user_id=current_user.id
    )
    db.add(key_obj)
    db.flush()

    # Audit log entry
    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="API_KEY_CREATED",
        details=f"Generated external API key '{payload.name}' ({key_obj.key_prefix})",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return {
        "success": True,
        "message": "API key generated successfully. Store this secret safely as it will not be shown again.",
        "api_key": {
            "id": key_obj.id,
            "name": key_obj.name,
            "secret_key": raw_token,
            "key_prefix": key_obj.key_prefix,
            "created_at": key_obj.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }
    }

@router.delete("/{key_id}")
def revoke_api_key(
    key_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Revoke and permanently delete an API key."""
    key_obj = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not key_obj:
        raise HTTPException(status_code=404, detail="API key not found")

    # Only owner or admin can revoke
    if key_obj.user_id != current_user.id and not current_user.is_admin():
        raise HTTPException(status_code=403, detail="Permission denied")

    name = key_obj.name
    prefix = key_obj.key_prefix
    db.delete(key_obj)

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="API_KEY_REVOKED",
        details=f"Revoked API key '{name}' ({prefix})",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return {
        "success": True,
        "message": f"API key '{name}' has been revoked."
    }
