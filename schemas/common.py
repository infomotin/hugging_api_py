from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    database: str
    app_name: str
    version: str

class StatsResponse(BaseModel):
    metrics: Dict[str, Any]
    roles_breakdown: Dict[str, int]
    recent_activity: List[Dict[str, Any]]

class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None
