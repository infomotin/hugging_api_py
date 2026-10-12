import uvicorn
from core.config import settings

if __name__ == "__main__":
    print(f"Starting server on http://{settings.HOST}:{settings.PORT}")
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=False)
