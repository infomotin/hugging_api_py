from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from core.config import settings
from core.dependencies import require_permission, require_admin
from models import User
from schemas.ai import (
    AiAnalyzeRequest, AiAnalyzeResponse,
    HFTokenRequest, HFTokenResponse,
    ModelDownloadRequest, ModelDownloadResponse,
    LocalModelOut, ModelTestRequest, ModelTestResponse
)
from AIModel.model_loader import model_manager
from AIModel.pipeline import AITextPipeline
from Utlility.logger import app_logger

router = APIRouter(prefix="/ai", tags=["Hugging Face & AI Pipelines"])

@router.post("/analyze", response_model=AiAnalyzeResponse)
def analyze_text(
    req: AiAnalyzeRequest,
    current_user: User = Depends(require_permission("use_ai_pipeline"))
):
    app_logger.info(f"AI analyze request from user '{current_user.username}' (len={len(req.text)})")
    res = AITextPipeline.analyze_sentiment(text=req.text, model_name=req.model_name)
    return AiAnalyzeResponse(**res)


@router.get("/info")
def get_ai_info(
    current_user: User = Depends(require_permission("use_ai_pipeline"))
):
    return AITextPipeline.get_info()


# --- Hugging Face API Configuration ---

@router.get("/hf-token/status", response_model=HFTokenResponse)
def get_hf_token_status(
    current_user: User = Depends(require_permission("view_dashboard"))
):
    status_info = model_manager.get_token_status()
    return HFTokenResponse(**status_info)


@router.post("/hf-token", response_model=HFTokenResponse)
def configure_hf_token(
    req: HFTokenRequest,
    current_user: User = Depends(require_admin)
):
    verification = model_manager.verify_hf_token(req.token)
    if verification.get("valid"):
        settings.update_hf_token(req.token)
        verification["configured"] = True
        app_logger.info(f"Admin '{current_user.username}' updated Hugging Face API token.")
        return HFTokenResponse(**verification)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=verification.get("message", "Invalid Hugging Face API token.")
        )


# --- Model Download & Catalog Management ---

@router.get("/models", response_model=List[LocalModelOut])
def list_local_models(
    current_user: User = Depends(require_permission("view_dashboard"))
):
    return model_manager.list_local_models()


@router.post("/download", response_model=ModelDownloadResponse)
def download_ai_model(
    req: ModelDownloadRequest,
    current_user: User = Depends(require_admin)
):
    app_logger.info(f"Admin '{current_user.username}' requested download of model: {req.model_id}")
    res = model_manager.download_model(model_id=req.model_id, token=req.token)
    if not res.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=res.get("message", "Model download failed.")
        )
    return ModelDownloadResponse(**res)


@router.post("/models/test", response_model=ModelTestResponse)
def test_local_model(
    req: ModelTestRequest,
    current_user: User = Depends(require_permission("use_ai_pipeline"))
):
    res = model_manager.test_local_model(slug=req.model_slug, text=req.text)
    return ModelTestResponse(**res)


@router.delete("/models/{model_slug}")
def delete_local_model(
    model_slug: str,
    current_user: User = Depends(require_admin)
):
    success = model_manager.delete_local_model(model_slug)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model_slug}' not found.")
    return {"message": f"Model '{model_slug}' deleted successfully from AIModel/model/"}
