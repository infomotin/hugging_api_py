import time
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user
from models import User, ChatMessage as DBChatMessage
from AIModel.chat_engine import chat_engine
from AIModel.model_loader import model_manager

router = APIRouter(prefix="", tags=["AI Chat Studio"])

class ChatMessageItem(BaseModel):
    role: str = "user"
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessageItem]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256
    session_id: Optional[str] = "default"

class SetActiveModelRequest(BaseModel):
    model_slug: str
    device_mode: Optional[str] = "cpu"
    gpu_layers: Optional[int] = 0
    vram_mb: Optional[int] = 0

@router.get("/chat/engine-status")
def get_engine_status(current_user: User = Depends(get_current_user)):
    """Returns runtime telemetry for the active AI model and GPU/V-RAM acceleration."""
    return {
        "status": "online",
        "engine": chat_engine.get_status()
    }

@router.get("/chat/history")
def get_chat_history(
    session_id: str = "default",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves stored conversation history from the PostgreSQL database for the current user."""
    history = db.query(DBChatMessage).filter(
        DBChatMessage.user_id == current_user.id
    ).order_by(DBChatMessage.created_at.asc()).all()

    return {
        "success": True,
        "messages": [m.to_dict() for m in history]
    }

@router.delete("/chat/history")
def clear_chat_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Clears and deletes all stored chat history for the current user from the database."""
    db.query(DBChatMessage).filter(DBChatMessage.user_id == current_user.id).delete()
    db.commit()
    return {
        "success": True,
        "message": "Chat history cleared from database."
    }

@router.post("/chat/active-model")
def set_active_chat_model(payload: SetActiveModelRequest, current_user: User = Depends(get_current_user)):
    """Switches the active inference model and configures CPU/GPU/V-RAM hardware allocations."""
    updated = chat_engine.set_active_model(
        slug=payload.model_slug,
        device_mode=payload.device_mode or "cpu",
        gpu_layers=payload.gpu_layers or 0,
        vram_mb=payload.vram_mb or 0
    )
    return {
        "success": True,
        "message": f"Active model set to '{payload.model_slug}' on {payload.device_mode.upper()}",
        "config": updated
    }

@router.post("/chat/completions")
def create_chat_completion(
    payload: ChatCompletionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Standard OpenAI-compatible Chat Completions API endpoint.
    Also stores all conversations into the PostgreSQL database!
    """
    if payload.model and payload.model != chat_engine.active_model_slug:
        local_models = [m["slug"] for m in model_manager.list_local_models()]
        if payload.model in local_models or model_manager.clean_model_slug(payload.model) in local_models:
            chat_engine.set_active_model(
                slug=payload.model,
                device_mode=chat_engine.device_mode,
                gpu_layers=chat_engine.gpu_layers,
                vram_mb=chat_engine.vram_mb
            )

    messages_dicts = [{"role": m.role, "content": m.content} for m in payload.messages]
    
    # Extract last user message
    last_user_prompt = ""
    for m in reversed(payload.messages):
        if m.role == "user":
            last_user_prompt = m.content
            break

    # 1. Save user prompt to database
    if last_user_prompt:
        user_record = DBChatMessage(
            user_id=current_user.id,
            session_id=payload.session_id or "default",
            role="user",
            content=last_user_prompt,
            model=chat_engine.active_model_slug,
            device=chat_engine.device_mode,
            tokens=len(last_user_prompt.split())
        )
        db.add(user_record)
        db.flush()

    # 2. Generate real AI reply
    result = chat_engine.generate_reply(
        messages=messages_dicts,
        temperature=payload.temperature or 0.7,
        max_tokens=payload.max_tokens or 256
    )
    reply_text = result.get("reply", "")

    # 3. Save assistant reply to database
    assistant_record = DBChatMessage(
        user_id=current_user.id,
        session_id=payload.session_id or "default",
        role="assistant",
        content=reply_text,
        model=chat_engine.active_model_slug,
        device=chat_engine.device_mode,
        tokens=result.get("usage", {}).get("completion_tokens", len(reply_text.split()))
    )
    db.add(assistant_record)
    db.commit()

    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    return {
        "id": cmpl_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": result.get("model") or "local-ai",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": result.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }),
        "hardware": {
            "device": result.get("device", "cpu"),
            "gpu_layers": chat_engine.gpu_layers,
            "vram_mb": chat_engine.vram_mb,
            "elapsed_seconds": result.get("elapsed_seconds", 0.0)
        }
    }

@router.get("/models")
def list_available_models(current_user: User = Depends(get_current_user)):
    """OpenAI-compatible models catalog list."""
    local_models = model_manager.list_local_models()
    data = []
    for m in local_models:
        data.append({
            "id": m["slug"],
            "object": "model",
            "created": 1728000000,
            "owned_by": "local-enterprise-node",
            "permission": [],
            "root": m["name"],
            "parent": None,
            "is_gguf": m.get("is_gguf", False),
            "size_mb": m.get("size_mb", 0)
        })
    return {
        "object": "list",
        "data": data
    }
