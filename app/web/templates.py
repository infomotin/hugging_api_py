import os
from fastapi.templating import Jinja2Templates
from fastapi import Request

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

ROUTE_MAP = {
    "index": "/",
    "auth.login": "/auth/login",
    "auth.register": "/auth/register",
    "auth.logout": "/auth/logout",
    "auth.profile": "/auth/profile",
    "dashboard.index": "/dashboard/",
    "dashboard.users": "/dashboard/users",
    "dashboard.roles": "/dashboard/roles",
    "dashboard.resources": "/dashboard/resources",
    "dashboard.audit_logs": "/dashboard/audit-logs",
    "dashboard.api_explorer": "/dashboard/api-explorer",
    "dashboard.ai_models": "/dashboard/ai-models",
    "dashboard.chat": "/dashboard/chat",
    "dashboard.api_keys": "/dashboard/api-keys",
    "dashboard.fine_tuning": "/dashboard/fine-tuning",
}

def render_template(template_name: str, request: Request, context: dict = None):
    # Compatibility aliases on request object
    if not hasattr(request, "args"):
        request.args = request.query_params
    if not hasattr(request, "endpoint"):
        request.endpoint = request.url.path

    ctx = {"request": request}
    if context:
        ctx.update(context)

    orig_url_for = ctx.get("url_for") or request.url_for
    def custom_url_for(name: str, **kwargs):
        if name in ROUTE_MAP:
            return ROUTE_MAP[name]
        if name == "static" and "filename" in kwargs:
            kwargs["path"] = kwargs.pop("filename")
        try:
            return str(orig_url_for(name, **kwargs))
        except Exception:
            if name == "static":
                return f"/static/{kwargs.get('path', '')}"
            return f"/{name.replace('.', '/')}"

    ctx["url_for"] = custom_url_for
    if "get_flashed_messages" not in ctx:
        ctx["get_flashed_messages"] = lambda with_categories=False: []

    return templates.TemplateResponse(request=request, name=template_name, context=ctx)
