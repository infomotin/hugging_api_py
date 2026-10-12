from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from core.database import get_db
from core.config import settings
from core.dependencies import require_permission
from models import User, Role, Resource, AuditLog
from schemas.common import HealthResponse, StatsResponse

router = APIRouter(tags=["Telemetry & System"])

@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        database=db_status,
        app_name=settings.PROJECT_NAME,
        version=settings.VERSION
    )


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission("view_dashboard"))
):
    total_users = db.query(User).count()
    active_users = db.query(User).filter_by(is_active=True).count()
    total_resources = db.query(Resource).count()
    roles = db.query(Role).all()

    roles_breakdown = {r.name: len(r.users) for r in roles}
    recent_logs = [l.to_dict() for l in db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(6).all()]

    return StatsResponse(
        metrics={
            "total_users": total_users,
            "active_users": active_users,
            "total_resources": total_resources,
            "roles_count": len(roles)
        },
        roles_breakdown=roles_breakdown,
        recent_activity=recent_logs
    )
