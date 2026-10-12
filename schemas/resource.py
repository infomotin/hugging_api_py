from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class ResourceBase(BaseModel):
    title: str
    description: Optional[str] = None
    category: Optional[str] = "General"
    status: Optional[str] = "Published"

class ResourceCreate(ResourceBase):
    pass

class ResourceUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None

class ResourceOut(ResourceBase):
    id: int
    author: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
