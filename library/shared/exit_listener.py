"""Graceful exit for long runs — deliberate triggers plus one safety guard, nothing global:

- Ctrl+C in the run's own window (SIGINT);
- a file named STOP in the working directory — what the dashboard's Stop button creates;
- the GPU running hotter than GPU_TEMPERATURE_LIMIT on two consecutive checks (checked
  every GPU_TEMPERATURE_CHECK_EVERY calls, via NVML, ~0.1 ms) — damage control that works
  whether or not the dashboard is open, and stops work below the card's own throttle point.

Any trigger sets `should_exit`; the training / generation loop checks it after every batch,
saves a resume checkpoint, and returns. The reason is written to queue/last_stop.json so
the dashboard can show it. A leftover STOP file is cleared when a run starts.
"""

import json
import os
import signal
from datetime import datetime

from library.shared.config import GPU_TEMPERATURE_CHECK_EVERY, GPU_TEMPERATURE_LIMIT

STOP_FILE = "STOP"
LAST_STOP_FILE = os.path.join("queue", "last_stop.json")


def read_gpu_temperature():
    """Current GPU temperature in °C, or None when NVML is unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        return None


class ExitListener:
    def __init__(self, temperature_reader=read_gpu_temperature):
        self.should_exit = False
        self.exit_reason = None
        self.temperature_reader = temperature_reader
        self.check_count = 0
        self.hot_streak = 0
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
        signal.signal(signal.SIGINT, self._signal_handler)
        print(f"Graceful exit: Ctrl+C here, the dashboard's Stop button (STOP file), "
              f"or GPU above {GPU_TEMPERATURE_LIMIT} C")

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
        if self.should_exit:
            return True
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
            self._trigger("STOP file (dashboard)")
            return True
        self.check_count += 1
        if self.check_count % GPU_TEMPERATURE_CHECK_EVERY == 0:
            temperature = self.temperature_reader()
            if temperature is not None and temperature >= GPU_TEMPERATURE_LIMIT:
                self.hot_streak += 1
                print(f"GPU at {temperature} C (limit {GPU_TEMPERATURE_LIMIT}), warning {self.hot_streak}/2")
                if self.hot_streak >= 2:
                    self._trigger(f"GPU temperature {temperature} C >= {GPU_TEMPERATURE_LIMIT} C")
            else:
                self.hot_streak = 0
        return self.should_exit

    def stop(self):
        pass
