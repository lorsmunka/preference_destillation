"""Graceful exit for long runs. Two deliberate triggers, nothing global:

- Ctrl+C in the run's own window (SIGINT), or
- a file named STOP in the working directory — what the dashboard's Stop button creates.

Either one sets `should_exit`; the training / generation loop checks it after every batch,
saves a resume checkpoint, and returns. A leftover STOP file is cleared when a run starts
so it can never kill the next one.
"""

import os
import signal

STOP_FILE = "STOP"


class ExitListener:
    def __init__(self):
        self.should_exit = False
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
        signal.signal(signal.SIGINT, self._signal_handler)
        print("Graceful exit: Ctrl+C here, or the dashboard's Stop button (creates a STOP file)")

    def _signal_handler(self, signum, frame):
        self.should_exit = True
        print("\nExit signal registered.")

    def check_exit(self) -> bool:
        if not self.should_exit and os.path.exists(STOP_FILE):
            self.should_exit = True
            os.remove(STOP_FILE)
            print("\nStop file found — exiting gracefully.")
        return self.should_exit

    def stop(self):
        pass
