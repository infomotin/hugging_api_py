from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class AiAnalyzeRequest(BaseModel):
    text: str
    task: Optional[str] = "sentiment-analysis"
    model_name: Optional[str] = "distilbert-base-uncased-finetuned-sst-2-english"

class AiAnalyzeResponse(BaseModel):
    task: str
    model: str
    input_text: str
    result: Any
    confidence_score: Optional[float] = None
    device: str
    status: str

# Hugging Face Token Schemas
class HFTokenRequest(BaseModel):
    token: str

class HFTokenResponse(BaseModel):
    valid: bool
    configured: bool
    username: Optional[str] = None
    email: Optional[str] = None
    orgs: Optional[List[str]] = []
    plan: Optional[str] = None
    message: str

# Model Download Schemas
class ModelDownloadRequest(BaseModel):
    model_id: str  # e.g. "prajjwal1/bert-tiny" or "distilbert/distilbert-base-uncased"
    token: Optional[str] = None

class ModelDownloadResponse(BaseModel):
    success: bool
    model_id: str
    slug: str
    path: Optional[str] = None
    size_mb: Optional[float] = None
    files_count: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    message: str

# Local Model Catalog Schemas
class LocalModelOut(BaseModel):
    slug: str
    model_id: str
    path: str
    relative_path: str
    size_mb: float
    file_count: int
    created_at: str
    is_ready: bool

class ModelTestRequest(BaseModel):
    model_slug: str
    text: str

class ModelTestResponse(BaseModel):
    success: bool
    model_slug: str
    input_text: str
    label: Optional[str] = None
    score: Optional[float] = None
    note: Optional[str] = None
    device: str
    error: Optional[str] = None
