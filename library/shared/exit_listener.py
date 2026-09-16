"""Graceful exit for long runs. Two deliberate triggers, nothing global:

- Ctrl+C in the run's own window (SIGINT), or
- a file named STOP in the working directory — what the dashboard's Stop button creates.

Either one sets `should_exit`; the training / generation loop checks it after every batch,
saves a resume checkpoint, and returns. The reason is written to queue/last_stop.json so the
dashboard can show it. A leftover STOP file is cleared when a run starts so it can never
kill the next one.
"""

import json
import os
import signal
from datetime import datetime

STOP_FILE = "STOP"
LAST_STOP_FILE = os.path.join("queue", "last_stop.json")


class ExitListener:
    def __init__(self):
        self.should_exit = False
        self.exit_reason = None
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
        signal.signal(signal.SIGINT, self._signal_handler)
        print("Graceful exit: Ctrl+C here, or the dashboard's Stop button (creates a STOP file)")

    def _signal_handler(self, signum, frame):
        self._trigger("Ctrl+C")

    def _trigger(self, reason: str):
        if self.should_exit:
            return
        self.should_exit = True
        self.exit_reason = reason
        print(f"\nExit requested ({reason}) — finishing the current batch and saving.")
        try:
            os.makedirs(os.path.dirname(LAST_STOP_FILE), exist_ok=True)
            with open(LAST_STOP_FILE, "w", encoding="utf-8") as file:
                json.dump({"reason": reason, "at": datetime.now().isoformat(timespec="seconds")}, file)
        except OSError:
            pass

    def check_exit(self) -> bool:
        if not self.should_exit and os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
            self._trigger("STOP file (dashboard)")
        return self.should_exit

    def stop(self):
        pass
