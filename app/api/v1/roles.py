from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import require_permission
from models import Role, Permission, User
from schemas.user import RoleOut, PermissionOut

router = APIRouter(prefix="/roles", tags=["Roles & Permissions"])

@router.get("", response_model=List[RoleOut])
def list_roles(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("manage_users"))
):
    roles = db.query(Role).all()
    res = []
    for r in roles:
        res.append(RoleOut(
            id=r.id,
            name=r.name,
            description=r.description,
            is_default=r.is_default,
            permissions=[p.name for p in r.permissions],
            user_count=len(r.users)
        ))
    return res

@router.get("/permissions", response_model=List[PermissionOut])
def list_permissions(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("manage_roles"))
):
    perms = db.query(Permission).order_by(Permission.name.asc()).all()
    return perms
