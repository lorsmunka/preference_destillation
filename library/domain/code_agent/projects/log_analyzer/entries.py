"""A LogEntry is one request from the web server's access log."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class LogEntry:
    ip: str
    time: datetime
    method: str
    path: str
    status: int
    size: int        # bytes sent back
    seconds: float   # how long the answer took

    @property
    def is_server_error(self):
        return self.status >= 500

    @property
    def status_class(self):
        """404 -> "4xx"."""
        return f"{self.status // 100}xx"

    @property
    def hour_label(self):
        """"2026-10-21 13:00" for any time from 13:00 to 13:59."""
        return self.time.strftime("%Y-%m-%d %H:00")
