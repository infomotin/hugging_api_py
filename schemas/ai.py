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

# Model Download Schemas (Supports GGUF)
class ModelDownloadRequest(BaseModel):
    model_id: str  # e.g. "Qwen/Qwen2.5-0.5B-Instruct-GGUF" or "prajjwal1/bert-tiny"
    filename: Optional[str] = None  # e.g. "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    is_gguf: Optional[bool] = False
    token: Optional[str] = None

class ModelDownloadResponse(BaseModel):
    success: bool
    model_id: str
    slug: str
    path: Optional[str] = None
    size_mb: Optional[float] = None
    files_count: Optional[int] = None
    is_gguf: Optional[bool] = False
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
    is_gguf: Optional[bool] = False
    gguf_files: Optional[List[str]] = []
    model_format: Optional[str] = "Standard"
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
    format: Optional[str] = None
    gguf_file: Optional[str] = None
    note: Optional[str] = None
    device: str
    error: Optional[str] = None

class HFModelSearchResult(BaseModel):
    id: str
    downloads: int
    likes: int
    pipeline_tag: str
    is_gguf: bool
