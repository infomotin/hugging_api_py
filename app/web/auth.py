from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Request, Form, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from core.database import get_db
from core.security import verify_password, hash_password, create_access_token
from core.dependencies import get_current_user_optional, get_current_user
from models import User, Role, AuditLog
from app.web.templates import render_template

router = APIRouter(prefix="/auth", tags=["Web Auth"])

@router.get("/login")
def login_page(request: Request, current_user: Optional[User] = Depends(get_current_user_optional)):
    if current_user and current_user.is_active:
        return RedirectResponse(url="/dashboard/", status_code=status.HTTP_302_FOUND)
    return render_template("auth/login.html", request, {"current_user": current_user})


@router.post("/login")
def login_action(
    request: Request,
    response: Response,
    identifier: str = Form(...),
    password: str = Form(...),
    remember: Optional[str] = Form(None),
    next: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    identifier = identifier.strip()
    user = db.query(User).filter(
        (User.username == identifier) | (User.email == identifier)
    ).first()

    if not user or not verify_password(password, user.password_hash):
        log = AuditLog(
            action="AUTH_FAILED_LOGIN",
            details=f"Failed login attempt for identifier: {identifier}",
            ip_address=request.client.host if request.client else "127.0.0.1"
        )
        db.add(log)
        db.commit()
        return render_template("auth/login.html", request, {
            "current_user": None,
            "get_flashed_messages": lambda with_categories=True: [("danger", "Invalid username/email or password.")]
        })

    if not user.is_active:
        return render_template("auth/login.html", request, {
            "current_user": None,
            "get_flashed_messages": lambda with_categories=True: [("danger", "Your account has been deactivated.")]
        })

    user.last_login = datetime.utcnow()
    token = create_access_token({"sub": user.username})

    log = AuditLog(
        user_id=user.id,
        username=user.username,
        action="AUTH_LOGIN",
        details=f"User {user.username} logged in via web portal.",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    target_url = "/dashboard/" if not next or not next.startswith("/") else next
    redirect_resp = RedirectResponse(url=target_url, status_code=status.HTTP_302_FOUND)
    redirect_resp.set_cookie(
        key="access_token",
        value=f"Bearer {token}",
        httponly=True,
        max_age=86400 * 7,
        samesite="lax"
    )
    return redirect_resp


@router.get("/register")
def register_page(request: Request, current_user: Optional[User] = Depends(get_current_user_optional)):
    if current_user and current_user.is_active:
        return RedirectResponse(url="/dashboard/", status_code=status.HTTP_302_FOUND)
    return render_template("auth/register.html", request, {"current_user": current_user})


@router.post("/register")
def register_action(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db)
):
    username = username.strip()
    email = email.strip().lower()

    if password != confirm_password:
        return render_template("auth/register.html", request, {
            "current_user": None,
            "get_flashed_messages": lambda with_categories=True: [("danger", "Passwords do not match.")]
        })

    if len(password) < 6:
        return render_template("auth/register.html", request, {
            "current_user": None,
            "get_flashed_messages": lambda with_categories=True: [("warning", "Password must be at least 6 characters.")]
        })

    if db.query(User).filter_by(username=username).first():
        return render_template("auth/register.html", request, {
            "current_user": None,
            "get_flashed_messages": lambda with_categories=True: [("warning", "Username is already taken.")]
        })

    if db.query(User).filter_by(email=email).first():
        return render_template("auth/register.html", request, {
            "current_user": None,
            "get_flashed_messages": lambda with_categories=True: [("warning", "Email is already registered.")]
        })

    default_role = db.query(Role).filter_by(is_default=True).first()
    if not default_role:
        default_role = db.query(Role).filter_by(name="User").first()

    new_user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
        role_id=default_role.id if default_role else 1,
        is_active=True
    )
    db.add(new_user)
    db.flush()

    token = create_access_token({"sub": new_user.username})

    log = AuditLog(
        user_id=new_user.id,
        username=new_user.username,
        action="USER_REGISTERED",
        details=f"New user registered: {new_user.username}",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    resp = RedirectResponse(url="/dashboard/", status_code=status.HTTP_302_FOUND)
    resp.set_cookie(
        key="access_token",
        value=f"Bearer {token}",
        httponly=True,
        max_age=86400 * 7,
        samesite="lax"
    )
    return resp


@router.get("/logout")
def logout_action(request: Request, current_user: Optional[User] = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    if current_user:
        log = AuditLog(
            user_id=current_user.id,
            username=current_user.username,
            action="AUTH_LOGOUT",
            details=f"User {current_user.username} signed out.",
            ip_address=request.client.host if request.client else "127.0.0.1"
        )
        db.add(log)
        db.commit()

    resp = RedirectResponse(url="/auth/login", status_code=status.HTTP_302_FOUND)
    resp.delete_cookie(key="access_token")
    return resp


@router.get("/profile")
def profile_page(request: Request, current_user: User = Depends(get_current_user)):
    return render_template("auth/profile.html", request, {"current_user": current_user})


@router.post("/profile")
def profile_update(
    request: Request,
    email: Optional[str] = Form(None),
    new_password: Optional[str] = Form(None),
    confirm_password: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    messages = []
    if email and email.strip().lower() != current_user.email:
        exist = db.query(User).filter_by(email=email.strip().lower()).first()
        if exist:
            messages.append(("danger", "Email already in use by another account."))
        else:
            current_user.email = email.strip().lower()
            messages.append(("success", "Email updated successfully."))

    if new_password:
        if new_password != confirm_password:
            messages.append(("danger", "Passwords do not match."))
        elif len(new_password) < 6:
            messages.append(("warning", "Password must be at least 6 characters."))
        else:
            current_user.password_hash = hash_password(new_password)
            messages.append(("success", "Password updated successfully."))

    db.commit()
    return render_template("auth/profile.html", request, {
        "current_user": current_user,
        "get_flashed_messages": lambda with_categories=True: messages
    })
