from fastapi import APIRouter
from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router
from app.api.v1.roles import router as roles_router
from app.api.v1.resources import router as resources_router
from app.api.v1.stats import router as stats_router
from app.api.v1.audit_logs import router as audit_logs_router
from app.api.v1.ai import router as ai_router
from app.api.v1.chat import router as chat_router
from app.api.v1.api_keys import router as api_keys_router
from app.api.v1.finetuning import router as finetuning_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(stats_router)
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(roles_router)
api_router.include_router(resources_router)
api_router.include_router(audit_logs_router)
api_router.include_router(ai_router)
api_router.include_router(chat_router)
api_router.include_router(api_keys_router)
api_router.include_router(finetuning_router)
