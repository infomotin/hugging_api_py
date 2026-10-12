from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user, require_permission
from models import Resource, User, AuditLog
from schemas.resource import ResourceOut, ResourceCreate

router = APIRouter(prefix="/resources", tags=["Resources"])

@router.get("", response_model=List[ResourceOut])
def list_resources(db: Session = Depends(get_db)):
    resources = db.query(Resource).order_by(Resource.created_at.desc()).all()
    out = []
    for r in resources:
        out.append(ResourceOut(
            id=r.id,
            title=r.title,
            description=r.description,
            category=r.category,
            status=r.status,
            author=r.author.username if r.author else "Unknown",
            created_at=r.created_at,
            updated_at=r.updated_at
        ))
    return out


@router.post("", response_model=ResourceOut, status_code=status.HTTP_201_CREATED)
def create_resource(
    req: ResourceCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("create_resources"))
):
    res = Resource(
        title=req.title,
        description=req.description,
        category=req.category or "General",
        status=req.status or "Published",
        created_by=current_user.id
    )
    db.add(res)
    db.flush()

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="RESOURCE_CREATED",
        details=f"Created resource: {res.title}",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()
    db.refresh(res)

    return ResourceOut(
        id=res.id,
        title=res.title,
        description=res.description,
        category=res.category,
        status=res.status,
        author=current_user.username,
        created_at=res.created_at,
        updated_at=res.updated_at
    )


@router.delete("/{res_id}")
def delete_resource(
    res_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    res = db.query(Resource).filter(Resource.id == res_id).first()
    if not res:
        raise HTTPException(status_code=404, detail="Resource not found")

    is_author = res.created_by == current_user.id
    has_perm = current_user.has_permission("delete_resources")

    if not (is_author or has_perm or current_user.is_admin()):
        raise HTTPException(status_code=403, detail="Not authorized to delete this resource")

    title = res.title
    db.delete(res)

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="RESOURCE_DELETED",
        details=f"Deleted resource: {title}",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return {"message": f"Resource '{title}' deleted successfully"}
