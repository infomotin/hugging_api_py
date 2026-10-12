# ⚡ FastAPI Enterprise RBAC & Hugging Face AI Platform

A modern, production-grade **FastAPI** web application featuring **Role-Based Access Control (RBAC)**, **PostgreSQL** database connectivity with SQLAlchemy ORM, **PyTorch & Hugging Face Transformers** integration, **Swagger OpenAPI** docs, and an executive administrative **dashboard**.

---

## 🏛️ Standard Architecture: `[app, core, models, schemas, main.py]`

```
hugging_api_py/
│
├── core/                         # Core infrastructure & system settings
│   ├── __init__.py
│   ├── config.py                 # Pydantic & environment settings (.env)
│   ├── database.py               # SQLAlchemy engine, SessionLocal, get_db generator
│   ├── security.py               # Password hashing & JWT generation/decoding
│   ├── dependencies.py           # RBAC dependency guards: require_role, require_permission
│   └── seed.py                   # Automatic database table provisioning & seed routines
│
├── models/                       # SQLAlchemy ORM Database Models
│   ├── __init__.py
│   ├── user.py                   # User, Role, Permission & association table
│   ├── resource.py               # Governed Resource & catalog entries
│   └── audit_log.py              # Security & operational audit logs
│
├── schemas/                      # Pydantic Schemas (Request / Response validation)
│   ├── __init__.py
│   ├── user.py                   # UserCreate, UserOut, RoleOut, PermissionOut, LoginRequest
│   ├── resource.py               # ResourceCreate, ResourceOut
│   ├── audit_log.py              # AuditLogOut
│   ├── common.py                 # HealthResponse, StatsResponse
│   └── ai.py                     # AiAnalyzeRequest, AiAnalyzeResponse (PyTorch/Transformers)
│
├── app/                          # Application Routers, Web Controllers & Assets
│   ├── __init__.py
│   ├── api/                      # RESTful API Endpoints (/api/v1)
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── router.py         # Aggregate API router
│   │       ├── auth.py           # /api/v1/auth/login, /api/v1/auth/register, /api/v1/auth/me
│   │       ├── users.py          # /api/v1/users (CRUD, Role assignment, Status toggle)
│   │       ├── roles.py          # /api/v1/roles, /api/v1/permissions
│   │       ├── resources.py      # /api/v1/resources (Catalog management)
│   │       ├── stats.py          # /api/v1/stats, /api/v1/health
│   │       ├── audit_logs.py     # /api/v1/audit-logs
│   │       └── ai.py             # /api/v1/ai/analyze (Hugging Face Transformers pipeline)
│   │
│   ├── web/                      # HTML Web Application Controllers
│   │   ├── __init__.py
│   │   ├── templates.py          # Jinja2 templates helper
│   │   ├── auth.py               # /auth/login, /auth/register, /auth/logout, /auth/profile
│   │   └── dashboard.py          # /dashboard/, /dashboard/users, /dashboard/roles, /dashboard/resources, /dashboard/audit-logs, /dashboard/api-explorer
│   │
│   ├── static/                   # Static CSS & JS Assets
│   │   ├── css/styles.css        # Modern glassmorphic dark styling & micro-animations
│   │   └── js/main.js            # Live API testing console & interactive UI helpers
│   │
│   └── templates/                # Jinja2 Templates
│       ├── base.html
│       ├── index.html
│       ├── auth/                 # login.html (1-click persona fill), register.html, profile.html
│       ├── dashboard/            # index.html, users.html, roles.html, resources.html, audit_logs.html, api_explorer.html
│       └── errors/               # 403.html (Access Denied), 404.html, 500.html
│
├── main.py                       # Root FastAPI Application Entry Point (Uvicorn runner)
├── requirements.txt              # Frozen dependencies (FastAPI, Uvicorn, Torch, Transformers, etc.)
├── .env                          # Local database & secret configurations
└── README.md
```

---

## ⚡ How to Run

1. **Activate the Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Start the Application**:
   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```
   *Or with Uvicorn directly:*
   ```powershell
   .\.venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 5050 --reload
   ```

3. **Open in Browser**:
   - **Web UI & Dashboard**: [http://127.0.0.1:5050](http://127.0.0.1:5050)
   - **Interactive Swagger Docs**: [http://127.0.0.1:5050/docs](http://127.0.0.1:5050/docs)
   - **ReDoc API Documentation**: [http://127.0.0.1:5050/redoc](http://127.0.0.1:5050/redoc)

---

## 👥 Pre-Configured Test Accounts

| Role | Username / Email | Password | Permissions Scope |
| :--- | :--- | :--- | :--- |
| **👑 Super Admin** | `admin` / `admin@example.com` | `Admin@123` | Full access: user management, role adjustments, deletion, audit logs, AI inference |
| **⚡ Manager** | `manager` / `manager@example.com` | `Manager@123` | Dashboard metrics, user inspection, resource creation, audit logs, AI inference |
| **👤 User** | `user` / `user@example.com` | `User@123` | Read catalog, create personal resources, self profile, AI inference |

---

## 🤖 Hugging Face & PyTorch AI Endpoint

The API includes a live AI pipeline endpoint powered by `torch` and `transformers`:

- **Endpoint**: `POST /api/v1/ai/analyze`
- **Guarded by**: `use_ai_pipeline` permission
- **Request Body**:
  ```json
  {
    "text": "Antigravity and FastAPI architecture is fast, flexible, and robust!",
    "task": "sentiment-analysis"
  }
  ```
- **Response**:
  ```json
  {
    "task": "sentiment-analysis",
    "model": "distilbert-base-uncased-finetuned-sst-2-english",
    "input_text": "Antigravity and FastAPI architecture is fast, flexible, and robust!",
    "result": {
      "label": "POSITIVE",
      "score": 0.9998
    },
    "confidence_score": 0.9998,
    "device": "cpu",
    "status": "success"
  }
  ```
