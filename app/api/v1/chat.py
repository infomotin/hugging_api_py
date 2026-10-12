import time
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from core.dependencies import get_current_user
from models import User
from AIModel.chat_engine import chat_engine
from AIModel.model_loader import model_manager

router = APIRouter(prefix="", tags=["AI Chat Studio"])

class ChatMessage(BaseModel):
    role: str = "user"
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256

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
def create_chat_completion(payload: ChatCompletionRequest, current_user: User = Depends(get_current_user)):
    """
    Standard OpenAI-compatible Chat Completions API endpoint.
    Accessible from outside via:
    - Authorization: Bearer sk-ent-...
    - X-API-Key: sk-ent-...
    - Session JWT Token
    """
    if payload.model and payload.model != chat_engine.active_model_slug:
        # If client explicitly specifies a downloaded model, temporarily switch or use it
        local_models = [m["slug"] for m in model_manager.list_local_models()]
        if payload.model in local_models or model_manager.clean_model_slug(payload.model) in local_models:
            chat_engine.set_active_model(
                slug=payload.model,
                device_mode=chat_engine.device_mode,
                gpu_layers=chat_engine.gpu_layers,
                vram_mb=chat_engine.vram_mb
            )

    messages_dicts = [{"role": m.role, "content": m.content} for m in payload.messages]
    result = chat_engine.generate_reply(
        messages=messages_dicts,
        temperature=payload.temperature or 0.7,
        max_tokens=payload.max_tokens or 256
    )

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
                    "content": result.get("reply", "")
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
    """
    OpenAI-compatible models catalog list.
    """
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
