import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class Settings:
    PROJECT_NAME: str = "Enterprise RBAC & AI Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-jwt-rbac-fastapi-token-key-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # PostgreSQL Database config
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASS: str = os.getenv("DB_PASS", "")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "flask_rbac_db")

    if DB_PASS:
        DATABASE_URI: str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    else:
        DATABASE_URI: str = f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    SQLITE_FALLBACK_URI: str = f"sqlite:///{os.path.join(BASE_DIR, 'app_fallback.db')}"
    PORT: int = int(os.getenv("PORT", 5050))
    HOST: str = os.getenv("HOST", "127.0.0.1")
    HF_TOKEN: str = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_API_TOKEN", ""))

    def update_hf_token(self, new_token: str):
        self.HF_TOKEN = new_token.strip()
        os.environ["HF_TOKEN"] = self.HF_TOKEN
        # Update .env file
        env_path = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if "HF_TOKEN=" in content:
                    lines = content.splitlines()
                    new_lines = [f"HF_TOKEN={self.HF_TOKEN}" if l.startswith("HF_TOKEN=") else l for l in lines]
                    content = "\n".join(new_lines)
                else:
                    content += f"\nHF_TOKEN={self.HF_TOKEN}\n"
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                print(f"[!] Error updating .env file: {e}")

settings = Settings()
