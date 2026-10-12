import os
import platform
import time
from datetime import datetime

START_TIME = time.time()

def get_system_telemetry() -> dict:
    uptime_seconds = int(time.time() - START_TIME)
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "uptime_seconds": uptime_seconds,
        "server_time": datetime.utcnow().isoformat(),
        "pid": os.getpid()
    }
