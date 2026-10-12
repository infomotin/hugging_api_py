from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class AuditLogOut(BaseModel):
    id: int
    user_id: Optional[int] = None
    username: Optional[str] = "System"
    action: str
    details: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True
