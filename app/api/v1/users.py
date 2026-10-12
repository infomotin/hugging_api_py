from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from core.database import get_db
from core.dependencies import get_current_user, require_permission, require_admin
from core.security import hash_password
from models import User, Role, AuditLog
from schemas.user import UserOut, UserCreate, RoleUpdateRequest

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("", response_model=List[UserOut])
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("manage_users"))
):
    users = db.query(User).order_by(User.id.asc()).all()
    result = []
    for u in users:
        result.append(UserOut(
            id=u.id,
            username=u.username,
            email=u.email,
            role=u.role.name if u.role else None,
            is_active=u.is_active,
            created_at=u.created_at,
            last_login=u.last_login
        ))
    return result


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    req: UserCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    if db.query(User).filter_by(username=req.username).first():
        raise HTTPException(status_code=409, detail="Username already exists")
    if db.query(User).filter_by(email=req.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    role_id = req.role_id
    if not role_id:
        default_role = db.query(Role).filter_by(is_default=True).first()
        role_id = default_role.id if default_role else 1

    new_user = User(
        username=req.username,
        email=req.email,
        password_hash=hash_password(req.password),
        role_id=role_id,
        is_active=req.is_active if req.is_active is not None else True
    )
    db.add(new_user)
    db.flush()

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="API_ADMIN_CREATE_USER",
        details=f"Admin created account: {new_user.username}",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()
    db.refresh(new_user)

    return UserOut(
        id=new_user.id,
        username=new_user.username,
        email=new_user.email,
        role=new_user.role.name if new_user.role else None,
        is_active=new_user.is_active,
        created_at=new_user.created_at,
        last_login=new_user.last_login
    )


@router.put("/{user_id}/role", response_model=UserOut)
def update_user_role(
    user_id: int,
    req: RoleUpdateRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    role = db.query(Role).filter(Role.id == req.role_id).first()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")

    old_role = user.role.name if user.role else "None"
    user.role_id = role.id

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="ROLE_UPDATED",
        details=f"Updated role for {user.username} from {old_role} to {role.name}",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()
    db.refresh(user)

    return UserOut(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role.name if user.role else None,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login
    )


@router.put("/{user_id}/status")
def toggle_user_status(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("manage_users"))
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot alter your own active status")

    user.is_active = not user.is_active
    status_str = "Activated" if user.is_active else "Deactivated"

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="STATUS_TOGGLED",
        details=f"{status_str} account for {user.username}",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return {"message": f"User status updated to {user.is_active}", "is_active": user.is_active}


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("delete_users"))
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    username = user.username
    db.delete(user)

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="USER_DELETED",
        details=f"Deleted user account: {username}",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return {"message": f"User {username} deleted successfully"}
