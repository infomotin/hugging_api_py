from datetime import datetime
from typing import Optional, Dict, Any

def format_datetime(dt: Optional[datetime], fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
    if not dt:
        return None
    return dt.strftime(fmt)

def standard_response(
    success: bool = True,
    message: str = "Operation completed",
    data: Optional[Any] = None
) -> Dict[str, Any]:
    return {
        "success": success,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
