import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import torch
import transformers
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user
from models import User, AuditLog
from AIModel.hf_studio_service import hf_studio_service, UPLOADED_DATASETS_DIR
from AIModel.model_loader import model_manager

router = APIRouter(prefix="/hf-studio", tags=["Hugging Face Central Workstation"])

class DatasetManipulateRequest(BaseModel):
    filename: str
    active_sheet: Optional[str] = None
    rename_columns: Optional[Dict[str, str]] = None
    drop_columns: Optional[List[str]] = None
    filter_query: Optional[str] = None
    new_sheet_name: Optional[str] = None

class CliRunRequest(BaseModel):
    command: str = Field(..., description="Hugging Face CLI, Python, or shell command to execute in COM mode")

class NotebookRunRequest(BaseModel):
    code: str = Field(..., description="Python/PyTorch code snippet to execute")
    session_id: Optional[str] = Field("default", description="Kernel session identifier")

@router.get("/system-specs")
def get_system_specs(current_user: User = Depends(get_current_user)):
    """Returns runtime telemetry for Hugging Face compute resources (CPU/CUDA, PyTorch, cache)."""
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Host Processor)"
    hf_token_status = model_manager.get_token_status()
    local_models = model_manager.list_local_models()

    return {
        "status": "online",
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": cuda_avail,
        "compute_device": device_name,
        "hf_authenticated": hf_token_status.get("valid", False),
        "hf_username": hf_token_status.get("username"),
        "local_models_count": len(local_models)
    }

# =========================================================================
# DATASET MANAGEMENT ROUTES
# =========================================================================
@router.get("/datasets/list")
def list_uploaded_datasets(current_user: User = Depends(get_current_user)):
    """List all user-uploaded datasets and spreadsheets stored on the server."""
    return {"datasets": hf_studio_service.list_uploaded_datasets()}

@router.get("/datasets/download/{filename}")
def download_dataset_file(filename: str, current_user: User = Depends(get_current_user)):
    """Downloads a dataset file from storage."""
    file_path = os.path.join(UPLOADED_DATASETS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

@router.delete("/datasets/{filename}")
def delete_dataset_file(filename: str, current_user: User = Depends(get_current_user)):
    """Deletes an uploaded dataset file from storage."""
    success = hf_studio_service.delete_dataset(filename)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    return {"success": True, "message": f"Dataset '{filename}' deleted."}

@router.post("/dataset/upload")
async def upload_dataset_file(
    file: UploadFile = File(...),
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Visual drag & drop handler for Excel (.xlsx, .xls), CSV, JSON, and Parquet datasets."""
    content = await file.read()
    try:
        parsed = hf_studio_service.parse_uploaded_dataset(file.filename, content)
        
        log = AuditLog(
            user_id=current_user.id,
            username=current_user.username,
            action="DATASET_UPLOADED",
            details=f"Uploaded dataset '{file.filename}' ({parsed['total_rows']} rows, {len(parsed['columns'])} columns)",
            ip_address=request.client.host if request and request.client else "127.0.0.1"
        )
        db.add(log)
        db.commit()

        return {
            "success": True,
            "message": f"Dataset '{file.filename}' parsed successfully.",
            "data": parsed
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/dataset/manipulate")
def manipulate_dataset_file(
    payload: DatasetManipulateRequest,
    current_user: User = Depends(get_current_user)
):
    """Applies column renames, sheet renaming, column dropping, and row filtering."""
    try:
        res = hf_studio_service.manipulate_dataset(
            filename=payload.filename,
            active_sheet=payload.active_sheet,
            rename_columns=payload.rename_columns,
            drop_columns=payload.drop_columns,
            filter_query=payload.filter_query,
            new_sheet_name=payload.new_sheet_name
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# =========================================================================
# REAL PROMPT COM / TERMINAL CLI ROUTE
# =========================================================================
@router.post("/cli/run")
def run_cli_command(
    payload: CliRunRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Executes command in real COM/Terminal prompt mode (HF CLI, Python, Pip, or Shell)."""
    res = hf_studio_service.run_hf_cli(payload.command)

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="HF_CLI_EXECUTED",
        details=f"Ran CLI: {payload.command[:80]} (exit code: {res['exit_code']})",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return res

# =========================================================================
# INTERACTIVE PYTORCH NOTEBOOK ROUTES
# =========================================================================
@router.post("/notebook/run")
def run_notebook_code(
    payload: NotebookRunRequest,
    current_user: User = Depends(get_current_user)
):
    """Executes interactive Python / PyTorch code cell in notebook kernel."""
    res = hf_studio_service.execute_notebook_cell(payload.code, payload.session_id or "default")
    return res

@router.post("/notebook/reset")
def reset_notebook_kernel(
    session_id: Optional[str] = Query("default"),
    current_user: User = Depends(get_current_user)
):
    """Resets variables in interactive notebook kernel."""
    hf_studio_service.reset_notebook_session(session_id)
    return {"success": True, "message": "Notebook kernel session reset."}
