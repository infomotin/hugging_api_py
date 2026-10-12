from typing import Optional
from fastapi import APIRouter, Depends, Request, Form, status, File, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from core.database import get_db
from core.security import hash_password
from core.dependencies import get_current_user_optional, get_current_user
from models import User, Role, Permission, Resource, AuditLog, ApiKey, ChatMessage
from app.web.templates import render_template

router = APIRouter(tags=["Web Dashboard"])

@router.get("/")
def home_page(request: Request, current_user: Optional[User] = Depends(get_current_user_optional)):
    if current_user and current_user.is_active:
        return RedirectResponse(url="/dashboard/", status_code=status.HTTP_302_FOUND)
    return render_template("index.html", request, {"current_user": current_user})


@router.get("/dashboard/")
def dashboard_home(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.has_permission("view_dashboard"):
        return RedirectResponse(url="/dashboard/resources", status_code=status.HTTP_302_FOUND)

    total_users = db.query(User).count()
    active_users = db.query(User).filter_by(is_active=True).count()
    roles = db.query(Role).all()
    total_resources = db.query(Resource).count()

    roles_stats = [{
        "name": r.name,
        "count": len(r.users),
        "percentage": round((len(r.users) / total_users * 100), 1) if total_users > 0 else 0
    } for r in roles]

    recent_activities = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(8).all()
    recent_users = db.query(User).order_by(User.created_at.desc()).limit(5).all()

    return render_template("dashboard/index.html", request, {
        "current_user": current_user,
        "total_users": total_users,
        "active_users": active_users,
        "total_roles": len(roles),
        "total_resources": total_resources,
        "roles_stats": roles_stats,
        "recent_activities": recent_activities,
        "recent_users": recent_users
    })


@router.get("/dashboard/users")
def users_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.has_permission("manage_users"):
        return render_template("errors/403.html", request, {
            "current_user": current_user,
            "required_permission": "manage_users"
        })

    all_users = db.query(User).order_by(User.id.asc()).all()
    all_roles = db.query(Role).all()
    return render_template("dashboard/users.html", request, {
        "current_user": current_user,
        "users": all_users,
        "roles": all_roles
    })


@router.post("/dashboard/users/create")
def create_user_form(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role_id: int = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_admin():
        return render_template("errors/403.html", request, {"current_user": current_user, "required_role": "Admin"})

    username = username.strip()
    email = email.strip().lower()

    if not db.query(User).filter_by(username=username).first() and not db.query(User).filter_by(email=email).first():
        new_u = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            role_id=role_id,
            is_active=True
        )
        db.add(new_u)
        db.flush()

        log = AuditLog(
            user_id=current_user.id,
            username=current_user.username,
            action="ADMIN_CREATE_USER",
            details=f"Created user {username}",
            ip_address=request.client.host if request.client else "127.0.0.1"
        )
        db.add(log)
        db.commit()

    return RedirectResponse(url="/dashboard/users", status_code=status.HTTP_302_FOUND)


@router.post("/dashboard/users/{user_id}/update-role")
def update_user_role_form(
    user_id: int,
    request: Request,
    role_id: int = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_admin():
        return render_template("errors/403.html", request, {"current_user": current_user, "required_role": "Admin"})

    user = db.query(User).filter(User.id == user_id).first()
    role = db.query(Role).filter(Role.id == role_id).first()
    if user and role:
        old_role = user.role.name if user.role else "None"
        user.role_id = role.id

        log = AuditLog(
            user_id=current_user.id,
            username=current_user.username,
            action="ROLE_UPDATED",
            details=f"Updated {user.username} role from {old_role} to {role.name}",
            ip_address=request.client.host if request.client else "127.0.0.1"
        )
        db.add(log)
        db.commit()

    return RedirectResponse(url="/dashboard/users", status_code=status.HTTP_302_FOUND)


@router.post("/dashboard/users/{user_id}/toggle-status")
def toggle_status_form(
    user_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.has_permission("manage_users"):
        return render_template("errors/403.html", request, {"current_user": current_user, "required_permission": "manage_users"})

    user = db.query(User).filter(User.id == user_id).first()
    if user and user.id != current_user.id:
        user.is_active = not user.is_active
        action_name = "Activated" if user.is_active else "Deactivated"

        log = AuditLog(
            user_id=current_user.id,
            username=current_user.username,
            action="STATUS_TOGGLED",
            details=f"{action_name} user {user.username}",
            ip_address=request.client.host if request.client else "127.0.0.1"
        )
        db.add(log)
        db.commit()

    return RedirectResponse(url="/dashboard/users", status_code=status.HTTP_302_FOUND)


@router.post("/dashboard/users/{user_id}/delete")
def delete_user_form(
    user_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.has_permission("delete_users"):
        return render_template("errors/403.html", request, {"current_user": current_user, "required_permission": "delete_users"})

    user = db.query(User).filter(User.id == user_id).first()
    if user and user.id != current_user.id:
        name = user.username
        db.delete(user)

        log = AuditLog(
            user_id=current_user.id,
            username=current_user.username,
            action="USER_DELETED",
            details=f"Deleted account {name}",
            ip_address=request.client.host if request.client else "127.0.0.1"
        )
        db.add(log)
        db.commit()

    return RedirectResponse(url="/dashboard/users", status_code=status.HTTP_302_FOUND)


@router.get("/dashboard/roles")
def roles_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.has_permission("manage_users"):
        return render_template("errors/403.html", request, {"current_user": current_user, "required_permission": "manage_users"})

    all_roles = db.query(Role).all()
    all_permissions = db.query(Permission).order_by(Permission.name.asc()).all()
    return render_template("dashboard/roles.html", request, {
        "current_user": current_user,
        "roles": all_roles,
        "permissions": all_permissions
    })


@router.get("/dashboard/resources")
def resources_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    all_resources = db.query(Resource).order_by(Resource.created_at.desc()).all()
    return render_template("dashboard/resources.html", request, {
        "current_user": current_user,
        "resources": all_resources
    })


@router.post("/dashboard/resources/create")
def create_resource_form(
    request: Request,
    title: str = Form(...),
    description: Optional[str] = Form(""),
    category: Optional[str] = Form("General"),
    status_val: Optional[str] = Form("Published"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.has_permission("create_resources"):
        return render_template("errors/403.html", request, {"current_user": current_user, "required_permission": "create_resources"})

    res = Resource(
        title=title.strip(),
        description=description.strip(),
        category=category,
        status=status_val,
        created_by=current_user.id
    )
    db.add(res)
    db.flush()

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="RESOURCE_CREATED",
        details=f"Created resource: {title}",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return RedirectResponse(url="/dashboard/resources", status_code=status.HTTP_302_FOUND)


@router.post("/dashboard/resources/{res_id}/delete")
def delete_resource_form(
    res_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    res = db.query(Resource).filter(Resource.id == res_id).first()
    if res and (res.created_by == current_user.id or current_user.has_permission("delete_resources") or current_user.is_admin()):
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

    return RedirectResponse(url="/dashboard/resources", status_code=status.HTTP_302_FOUND)


@router.get("/dashboard/audit-logs")
def audit_logs_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.has_permission("view_audit_logs"):
        return render_template("errors/403.html", request, {"current_user": current_user, "required_permission": "view_audit_logs"})

    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(100).all()
    return render_template("dashboard/audit_logs.html", request, {
        "current_user": current_user,
        "logs": logs
    })


@router.get("/dashboard/api-explorer")
def api_explorer_page(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    return render_template("dashboard/api_explorer.html", request, {
        "current_user": current_user
    })


@router.get("/dashboard/ai-models")
def ai_models_page(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    from AIModel.model_loader import model_manager
    hf_status = model_manager.get_token_status()
    local_models = model_manager.list_local_models()
    telemetry = model_manager.get_telemetry()

    return render_template("dashboard/ai_models.html", request, {
        "current_user": current_user,
        "hf_status": hf_status,
        "local_models": local_models,
        "telemetry": telemetry
    })


@router.post("/dashboard/ai-models/hf-token")
def update_hf_token_form(
    request: Request,
    token: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    from AIModel.model_loader import model_manager
    from core.config import settings
    verification = model_manager.verify_hf_token(token)
    if verification.get("valid"):
        settings.update_hf_token(token)
        log = AuditLog(
            user_id=current_user.id,
            username=current_user.username,
            action="HF_TOKEN_UPDATED",
            details=f"Connected Hugging Face account: @{verification.get('username')}",
            ip_address=request.client.host if request.client else "127.0.0.1"
        )
        return RedirectResponse(url="/dashboard/ai-models", status_code=status.HTTP_302_FOUND)
    
    return RedirectResponse(url="/dashboard/ai-models?error=invalid_token", status_code=status.HTTP_302_FOUND)


@router.post("/dashboard/ai-models/download")
def download_model_form(
    request: Request,
    model_id: str = Form(...),
    filename: Optional[str] = Form(None),
    is_gguf: Optional[bool] = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from AIModel.model_loader import model_manager
    res = model_manager.download_model(
        model_id=model_id,
        filename=filename,
        is_gguf=is_gguf
    )

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="HF_MODEL_DOWNLOADED" if res.get("success") else "HF_DOWNLOAD_FAILED",
        details=res.get("message"),
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return RedirectResponse(url="/dashboard/ai-models", status_code=status.HTTP_302_FOUND)


@router.post("/dashboard/ai-models/upload")
async def upload_model_form(
    request: Request,
    model_file: UploadFile = File(...),
    custom_name: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from AIModel.model_loader import model_manager
    content = await model_file.read()
    res = model_manager.save_uploaded_model(
        file_name=model_file.filename,
        file_bytes=content,
        custom_name=custom_name
    )

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="LOCAL_MODEL_UPLOADED" if res.get("success") else "UPLOAD_FAILED",
        details=res.get("message"),
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return RedirectResponse(url="/dashboard/ai-models", status_code=status.HTTP_302_FOUND)


@router.post("/dashboard/ai-models/{model_slug}/delete")
def delete_local_model_form(
    model_slug: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from AIModel.model_loader import model_manager
    success = model_manager.delete_local_model(model_slug)
    if success:
        log = AuditLog(
            user_id=current_user.id,
            username=current_user.username,
            action="LOCAL_MODEL_DELETED",
            details=f"Removed model '{model_slug}' from AIModel/model/",
            ip_address=request.client.host if request.client else "127.0.0.1"
        )
        db.add(log)
        db.commit()

    return RedirectResponse(url="/dashboard/ai-models", status_code=status.HTTP_302_FOUND)


@router.get("/dashboard/chat")
def chat_studio_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from AIModel.model_loader import model_manager
    from AIModel.chat_engine import chat_engine

    local_models = model_manager.list_local_models()
    engine_status = chat_engine.get_status()
    api_keys = db.query(ApiKey).filter(ApiKey.user_id == current_user.id, ApiKey.is_active == True).all()
    chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == current_user.id).order_by(ChatMessage.created_at.asc()).all()

    return render_template("dashboard/chat.html", request, {
        "current_user": current_user,
        "local_models": local_models,
        "engine_status": engine_status,
        "api_keys": api_keys,
        "chat_history": chat_history
    })


@router.post("/dashboard/chat/active-model")
def set_active_chat_model_web(
    request: Request,
    model_slug: str = Form(...),
    device_mode: str = Form("cpu"),
    gpu_layers: int = Form(0),
    vram_mb: int = Form(0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from AIModel.chat_engine import chat_engine
    chat_engine.set_active_model(
        slug=model_slug,
        device_mode=device_mode,
        gpu_layers=gpu_layers,
        vram_mb=vram_mb
    )

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="ACTIVE_MODEL_UPDATED",
        details=f"Active model switched to {model_slug} on {device_mode.upper()} (GPU layers: {gpu_layers}, V-RAM: {vram_mb}MB)",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return RedirectResponse(url="/dashboard/chat?updated=1", status_code=status.HTTP_302_FOUND)


@router.get("/dashboard/api-keys")
def api_keys_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    keys = db.query(ApiKey).filter(ApiKey.user_id == current_user.id).order_by(ApiKey.created_at.desc()).all()
    return render_template("dashboard/api_keys.html", request, {
        "current_user": current_user,
        "api_keys": keys,
        "new_token": request.query_params.get("new_token")
    })


@router.post("/dashboard/api-keys/create")
def create_api_key_web(
    request: Request,
    name: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    raw_token, key_obj = ApiKey.generate_key_pair(
        name=name.strip(),
        user_id=current_user.id
    )
    db.add(key_obj)
    db.flush()

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="API_KEY_CREATED",
        details=f"Generated key '{name}' ({key_obj.key_prefix})",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(log)
    db.commit()

    return RedirectResponse(url=f"/dashboard/api-keys?new_token={raw_token}", status_code=status.HTTP_302_FOUND)


@router.post("/dashboard/api-keys/{key_id}/delete")
def delete_api_key_web(
    key_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    key_obj = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if key_obj and (key_obj.user_id == current_user.id or current_user.is_admin()):
        name = key_obj.name
        prefix = key_obj.key_prefix
        db.delete(key_obj)

        log = AuditLog(
            user_id=current_user.id,
            username=current_user.username,
            action="API_KEY_REVOKED",
            details=f"Revoked API key '{name}' ({prefix})",
            ip_address=request.client.host if request.client else "127.0.0.1"
        )
        db.add(log)
        db.commit()

    return RedirectResponse(url="/dashboard/api-keys", status_code=status.HTTP_302_FOUND)


@router.get("/dashboard/fine-tuning")
def fine_tuning_studio_page(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    from AIModel.model_loader import model_manager
    from AIModel.finetune_engine import finetune_engine
    from AIModel.dataset_hub import POPULAR_HF_DATASETS

    local_models = model_manager.list_local_models()
    theoretical_ref = finetune_engine.get_theoretical_reference()
    jobs = finetune_engine.list_jobs()

    finetuned_models = [m for m in local_models if "finetuned" in m["slug"].lower() or "ft" in m["slug"].lower()]
    default_ft_slug = finetuned_models[0]["slug"] if finetuned_models else (local_models[0]["slug"] if local_models else "finetuned-extractbench-v1")
    initial_eval = finetune_engine.get_model_evaluation(default_ft_slug)

    return render_template("dashboard/fine_tuning.html", request, {
        "current_user": current_user,
        "local_models": local_models,
        "finetuned_models": finetuned_models,
        "initial_eval": initial_eval,
        "theoretical_ref": theoretical_ref,
        "popular_datasets": POPULAR_HF_DATASETS,
        "recent_jobs": jobs
    })


@router.get("/dashboard/hf-studio")
def hf_studio_page(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    import torch
    import transformers
    from AIModel.model_loader import model_manager
    from AIModel.dataset_hub import POPULAR_HF_DATASETS
    from AIModel.hf_studio_service import hf_studio_service

    specs = {
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "compute_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (Host Processor)"
    }
    hf_status = model_manager.get_token_status()
    local_models = model_manager.list_local_models()
    uploaded_datasets = hf_studio_service.list_uploaded_datasets()

    return render_template("dashboard/hf_studio.html", request, {
        "current_user": current_user,
        "specs": specs,
        "hf_status": hf_status,
        "local_models": local_models,
        "popular_datasets": POPULAR_HF_DATASETS,
        "uploaded_datasets": uploaded_datasets,
        "default_prompt": f"HF-STUDIO {hf_studio_service.workspace_dir}>"
    })

