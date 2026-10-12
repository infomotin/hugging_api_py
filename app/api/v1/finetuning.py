from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user
from models import User, AuditLog
from AIModel.finetune_engine import finetune_engine
from AIModel.dataset_hub import dataset_hub
from AIModel.model_loader import model_manager

router = APIRouter(prefix="/fine-tuning", tags=["Model Fine-Tuning Studio"])

class StartFineTuningRequest(BaseModel):
    base_model_slug: str = Field(..., description="Target downloaded model from AIModel/model/")
    output_model_name: str = Field(..., description="Unique name for the fine-tuned artifact")
    dataset_id: str = Field("llamaindex/ExtractBench", description="Hugging Face dataset identifier")
    dataset_split: Optional[str] = Field("short", description="Dataset split (e.g. short, medium, long or train)")
    input_column: Optional[str] = Field("data_schema", description="Input feature column")
    target_column: Optional[str] = Field("expected_output", description="Target label/output column")
    prompt_template: Optional[str] = Field("[EXTRACT SCHEMA]: {input} -> [TARGET]: {target}")
    method: Optional[str] = Field("lora", description="'lora', 'qlora', 'prefix', or 'full'")
    learning_rate: Optional[float] = Field(5e-5, description="Initial learning rate")
    epochs: Optional[int] = Field(3, ge=1, le=20, description="Number of training epochs")
    batch_size: Optional[int] = Field(2, ge=1, le=64, description="Training batch size")
    lora_r: Optional[int] = Field(8, ge=2, le=64, description="LoRA rank dimension")
    lora_alpha: Optional[int] = Field(16, ge=4, le=128, description="LoRA scaling factor")
    lora_dropout: Optional[float] = Field(0.05, ge=0.0, le=0.5, description="LoRA dropout rate")
    optimizer_name: Optional[str] = Field("AdamW")
    scheduler_name: Optional[str] = Field("cosine")
    device: Optional[str] = Field("cpu")
    warmup_ratio: Optional[float] = Field(0.1, ge=0.0, le=0.5)
    weight_decay: Optional[float] = Field(0.01, ge=0.0, le=0.1)
    max_grad_norm: Optional[float] = Field(1.0, ge=0.1, le=10.0)
    eval_split: Optional[float] = Field(0.15, ge=0.05, le=0.4)

class CompareInferenceRequest(BaseModel):
    base_model_slug: str = Field(..., description="Baseline model name (e.g. gpt2 or prajjwal1--bert-tiny)")
    finetuned_model_slug: str = Field(..., description="Fine-tuned model name (e.g. finetuned-extractbench-v1)")
    prompt: str = Field(..., description="Prompt to evaluate side-by-side")
    max_tokens: Optional[int] = Field(120, ge=10, le=512)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)

@router.get("/theoretical-reference")
def get_theoretical_reference(current_user: User = Depends(get_current_user)):
    """Returns mathematical formulations, LaTeX representations, and theoretical foundation notes."""
    return finetune_engine.get_theoretical_reference()

@router.get("/evaluation/{model_slug}")
def get_model_evaluation(
    model_slug: str,
    current_user: User = Depends(get_current_user)
):
    """Fetch rigorous Before vs After mathematical evaluation metrics for a fine-tuned model."""
    eval_data = finetune_engine.get_model_evaluation(model_slug)
    return {"evaluation": eval_data}

@router.post("/compare-inference")
def compare_model_inference(
    payload: CompareInferenceRequest,
    current_user: User = Depends(get_current_user)
):
    """Run live side-by-side prompt execution comparing base model against fine-tuned model."""
    try:
        comparison = finetune_engine.compare_inference(
            base_model_slug=payload.base_model_slug,
            finetuned_model_slug=payload.finetuned_model_slug,
            prompt=payload.prompt,
            max_tokens=payload.max_tokens or 120,
            temperature=payload.temperature or 0.7
        )
        return {"success": True, "comparison": comparison}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/datasets/search")
def search_hf_datasets(
    query: str = Query("", description="Search term for Hugging Face datasets"),
    limit: int = Query(12, ge=1, le=30),
    current_user: User = Depends(get_current_user)
):
    """Search Hugging Face datasets hub or list curated extraction benchmarks."""
    results = dataset_hub.search_datasets(query=query, limit=limit)
    return {"results": results}

@router.get("/datasets/splits")
def get_dataset_splits(
    dataset_id: str = Query(..., description="Hugging Face dataset repository ID"),
    current_user: User = Depends(get_current_user)
):
    """Fetch configs and splits for a dataset (e.g. short, medium, long)."""
    splits = dataset_hub.fetch_dataset_splits(dataset_id=dataset_id)
    return {"dataset": dataset_id, "splits": splits}

@router.get("/datasets/rows")
def get_dataset_rows(
    dataset_id: str = Query("llamaindex/ExtractBench"),
    config: str = Query("default"),
    split: str = Query("short"),
    offset: int = Query(0, ge=0),
    length: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user)
):
    """Fetch live schema and sample rows for the Dataface viewer."""
    data = dataset_hub.fetch_dataset_rows(
        dataset_id=dataset_id,
        config=config,
        split=split,
        offset=offset,
        length=length
    )
    return data

@router.post("/start", status_code=status.HTTP_202_ACCEPTED)
def start_fine_tuning_job(
    payload: StartFineTuningRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Launch an asynchronous background fine-tuning job with live monitoring telemetry and mathematical tracking."""
    local_slugs = [m["slug"] for m in model_manager.list_local_models()]
    if payload.base_model_slug not in local_slugs and model_manager.clean_model_slug(payload.base_model_slug) not in local_slugs:
        raise HTTPException(
            status_code=400,
            detail=f"Base model '{payload.base_model_slug}' is not found in downloaded local models. Available: {local_slugs}"
        )

    job_info = finetune_engine.start_fine_tuning(
        base_model_slug=payload.base_model_slug,
        output_model_name=payload.output_model_name,
        dataset_id=payload.dataset_id,
        dataset_split=payload.dataset_split or "short",
        input_column=payload.input_column or "data_schema",
        target_column=payload.target_column or "expected_output",
        prompt_template=payload.prompt_template or "[EXTRACT SCHEMA]: {input} -> [TARGET]: {target}",
        method=payload.method or "lora",
        learning_rate=payload.learning_rate or 5e-5,
        epochs=payload.epochs or 3,
        batch_size=payload.batch_size or 2,
        lora_r=payload.lora_r or 8,
        lora_alpha=payload.lora_alpha or 16,
        lora_dropout=payload.lora_dropout or 0.05,
        optimizer_name=payload.optimizer_name or "AdamW",
        scheduler_name=payload.scheduler_name or "cosine",
        device=payload.device or "cpu",
        warmup_ratio=payload.warmup_ratio or 0.1,
        weight_decay=payload.weight_decay or 0.01,
        max_grad_norm=payload.max_grad_norm or 1.0,
        eval_split=payload.eval_split or 0.15
    )

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="FINE_TUNING_STARTED",
        details=f"Launched {payload.method.upper()} fine-tuning on '{payload.base_model_slug}' using {payload.dataset_id} (Job: {job_info['job_id']})",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return {
        "success": True,
        "message": f"Fine-tuning job '{job_info['job_id']}' launched successfully in background.",
        "job": job_info
    }

@router.get("/jobs/{job_id}")
def get_job_progress(job_id: str, current_user: User = Depends(get_current_user)):
    """Fetch real-time training progress, loss curves, and terminal logs."""
    job = finetune_engine.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/jobs/{job_id}/evaluation")
def get_job_evaluation(job_id: str, current_user: User = Depends(get_current_user)):
    """Fetch before-vs-after evaluation report for a completed job."""
    job = finetune_engine.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"evaluation": job.get("evaluation_comparison")}

@router.post("/jobs/{job_id}/stop")
def stop_fine_tuning_job(job_id: str, current_user: User = Depends(get_current_user)):
    """Stop an ongoing training job."""
    stopped = finetune_engine.stop_job(job_id)
    if not stopped:
        raise HTTPException(status_code=400, detail="Job is not currently running or cannot be stopped.")
    return {"success": True, "message": f"Job '{job_id}' stopped."}

@router.get("/jobs")
def list_all_jobs(current_user: User = Depends(get_current_user)):
    """List all recent fine-tuning jobs."""
    return {"jobs": finetune_engine.list_jobs()}
