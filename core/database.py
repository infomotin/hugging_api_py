from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from core.config import settings

Base = declarative_base()

def get_engine():
    try:
        # Test postgres engine
        eng = create_engine(
            settings.DATABASE_URI,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3} if "postgresql" in settings.DATABASE_URI else {}
        )
        with eng.connect() as conn:
            pass
        return eng
    except Exception as e:
        print(f"[!] PostgreSQL connect notice ({e}). Falling back to SQLite...")
        return create_engine(
            settings.SQLITE_FALLBACK_URI,
            connect_args={"check_same_thread": False}
        )

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
