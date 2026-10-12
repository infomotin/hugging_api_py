from core.database import engine, Base, SessionLocal
from models import Permission, Role, User, Resource, AuditLog
from core.security import hash_password

def seed_database():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # 1. Permissions
        permissions_data = [
            ("view_dashboard", "Access the backend administration dashboard"),
            ("manage_users", "Create, edit, and update user accounts"),
            ("delete_users", "Permanently remove user accounts"),
            ("manage_roles", "Configure role permissions and access matrices"),
            ("create_resources", "Create new resources and posts"),
            ("edit_resources", "Update and publish existing resources"),
            ("delete_resources", "Remove resources from the system"),
            ("view_audit_logs", "Inspect security audit logs and user activities"),
            ("use_ai_pipeline", "Execute Hugging Face ML/NLP transformer models"),
        ]

        perm_map = {}
        for name, desc in permissions_data:
            perm = db.query(Permission).filter_by(name=name).first()
            if not perm:
                perm = Permission(name=name, description=desc)
                db.add(perm)
            perm_map[name] = perm
        db.commit()

        # 2. Roles
        admin_role = db.query(Role).filter_by(name="Admin").first()
        if not admin_role:
            admin_role = Role(name="Admin", description="Complete system control and security administration", is_default=False)
            db.add(admin_role)
        admin_role.permissions = list(perm_map.values())

        manager_role = db.query(Role).filter_by(name="Manager").first()
        if not manager_role:
            manager_role = Role(name="Manager", description="Content and user operations manager", is_default=False)
            db.add(manager_role)
        manager_role.permissions = [
            perm_map["view_dashboard"],
            perm_map["manage_users"],
            perm_map["create_resources"],
            perm_map["edit_resources"],
            perm_map["view_audit_logs"],
            perm_map["use_ai_pipeline"],
        ]

        user_role = db.query(Role).filter_by(name="User").first()
        if not user_role:
            user_role = Role(name="User", description="Standard user with self-service & catalog read access", is_default=True)
            db.add(user_role)
        user_role.permissions = [
            perm_map["create_resources"],
            perm_map["use_ai_pipeline"],
        ]
        db.commit()

        # 3. Demo Users
        users_data = [
            ("admin", "admin@example.com", "Admin@123", admin_role),
            ("manager", "manager@example.com", "Manager@123", manager_role),
            ("user", "user@example.com", "User@123", user_role),
        ]

        for username, email, pwd, role in users_data:
            existing = db.query(User).filter_by(username=username).first()
            if not existing:
                u = User(
                    username=username,
                    email=email,
                    role_id=role.id,
                    is_active=True,
                    password_hash=hash_password(pwd)
                )
                db.add(u)
        db.commit()

        # 4. Sample Resources & Initial Log
        admin_user = db.query(User).filter_by(username="admin").first()
        if admin_user and db.query(Resource).count() == 0:
            sample_resources = [
                Resource(
                    title="Q4 Security & Role Access Audit Report",
                    description="Comprehensive audit of all role assignments, API tokens, and access matrix boundaries across microservices.",
                    category="Security",
                    status="Published",
                    created_by=admin_user.id
                ),
                Resource(
                    title="FastAPI + PyTorch Pipeline Guide 2026",
                    description="Asynchronous serving architecture for Hugging Face transformer pipelines with token authentication.",
                    category="Engineering",
                    status="Published",
                    created_by=admin_user.id
                ),
                Resource(
                    title="Enterprise RBAC Operational Playbook",
                    description="Step-by-step procedures for granting permissions, revoking tokens, and inspecting security trails.",
                    category="Operations",
                    status="Published",
                    created_by=admin_user.id
                )
            ]
            db.bulk_save_objects(sample_resources)

            log = AuditLog(
                user_id=admin_user.id,
                username=admin_user.username,
                action="SYSTEM_INIT",
                details="FastAPI platform initialized with RBAC roles, permissions, and PyTorch capability.",
                ip_address="127.0.0.1"
            )
            db.add(log)
            db.commit()
            print("[*] Seeded demo users, roles, resources, and audit log.")

    except Exception as e:
        db.rollback()
        print(f"[!] Seed error: {e}")
    finally:
        db.close()
