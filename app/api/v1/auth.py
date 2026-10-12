from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from core.database import get_db
from core.security import verify_password, hash_password, create_access_token
from core.dependencies import get_current_user
from models import User, Role, AuditLog
from schemas.user import LoginRequest, TokenResponse, UserCreate, UserOut

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login", response_model=TokenResponse)
def api_login(req: LoginRequest, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        (User.username == req.username) | (User.email == req.username)
    ).first()

    if not user or not verify_password(req.password, user.password_hash):
        # Log failed attempt
        log = AuditLog(
            action="API_FAILED_LOGIN",
            details=f"Failed API login attempt for username: {req.username}",
            ip_address=request.client.host if request.client else "127.0.0.1"
        )
        db.add(log)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )

    user.last_login = datetime.utcnow()
    token = create_access_token({"sub": user.username})

    log = AuditLog(
        user_id=user.id,
        username=user.username,
        action="API_LOGIN",
        details=f"User {user.username} authenticated via REST API.",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    perms = [p.name for p in user.role.permissions] if user.role else []
    user_out = UserOut(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role.name if user.role else None,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=user_out,
        permissions=perms
    )


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def api_register(req: UserCreate, request: Request, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username is already taken")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email is already registered")

    default_role = db.query(Role).filter_by(is_default=True).first()
    if not default_role:
        default_role = db.query(Role).filter_by(name="User").first()

    new_user = User(
        username=req.username,
        email=req.email,
        password_hash=hash_password(req.password),
        role_id=default_role.id if default_role else 1,
        is_active=True
    )
    db.add(new_user)
    db.flush()

    log = AuditLog(
        user_id=new_user.id,
        username=new_user.username,
        action="USER_REGISTERED",
        details=f"New user registered via API: {new_user.username}",
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


@router.get("/me")
def api_me(current_user: User = Depends(get_current_user)):
    return {
        "user": current_user.to_dict(),
        "role": current_user.role.name if current_user.role else None,
        "permissions": [p.name for p in current_user.role.permissions] if current_user.role else []
    }
