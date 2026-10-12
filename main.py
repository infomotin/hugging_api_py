import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from core.config import settings
from core.seed import seed_database
from app.api.v1.router import api_router
from app.web.auth import router as web_auth_router
from app.web.dashboard import router as web_dashboard_router
from app.web.templates import render_template
from Utlility.logger import app_logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("Initializing Enterprise RBAC & AI Platform...")
    seed_database()
    app_logger.info("Database and RBAC seed tables verified.")
    yield
    app_logger.info("Platform shutting down gracefully.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Role-Based Access Control Platform with FastAPI, PostgreSQL, and PyTorch AI Pipelines",
    lifespan=lifespan
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "app", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount API & Web Routers
app.include_router(api_router)
app.include_router(web_auth_router)
app.include_router(web_dashboard_router)

# Custom Error Handlers
@app.exception_handler(401)
async def unauthorized_exception_handler(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=401, content={"error": "Unauthorized", "detail": str(exc.detail if hasattr(exc, "detail") else "Authentication required")})
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/auth/login?next={request.url.path}", status_code=302)

@app.exception_handler(403)
async def forbidden_exception_handler(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=403, content={"error": "Forbidden", "detail": str(exc.detail if hasattr(exc, "detail") else exc)})
    return render_template("errors/403.html", request, {"current_user": None})

@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=404, content={"error": "Not Found", "detail": "Endpoint not found"})
    return render_template("errors/404.html", request, {"current_user": None})

if __name__ == "__main__":
    print("\n=======================================================")
    print(f"  {settings.PROJECT_NAME} Running")
    print(f"  URL: http://{settings.HOST}:{settings.PORT}")
    print(f"  Interactive Swagger Docs: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"  ReDoc OpenAPI:            http://{settings.HOST}:{settings.PORT}/redoc")
    print(f"  Demo Accounts:")
    print(f"    Admin:   admin@example.com   / Admin@123")
    print(f"    Manager: manager@example.com / Manager@123")
    print(f"    User:    user@example.com    / User@123")
    print("=======================================================\n")
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=False)
