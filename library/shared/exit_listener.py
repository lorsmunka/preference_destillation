import os
import signal
import sys

STOP_FILE = "STOP"   # create this file in the working directory to request a graceful exit

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class ExitListener:
    def __init__(self, exit_key='%'):
        self.should_exit = False
        self.exit_key = exit_key
        self.listener = None
        if os.path.exists(STOP_FILE):   # a leftover from an earlier stop must not kill this run
            os.remove(STOP_FILE)

        if PYNPUT_AVAILABLE:
            try:
                self.listener = keyboard.Listener(on_press=self._on_press)
                self.listener.start()
            except Exception:
                self._setup_signal_handler()
        else:
            self._setup_signal_handler()

    def _setup_signal_handler(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        print("Headless mode: press Ctrl+C for graceful exit")

    def _signal_handler(self, signum, frame):
        self.should_exit = True
        print("\nExit signal registered.")

    def _on_press(self, key):
        try:
            if key.char == self.exit_key:
                self.should_exit = True
                print("Exit key registered.")
        except AttributeError:
            pass

    def check_exit(self):
        if not self.should_exit and os.path.exists(STOP_FILE):
            self.should_exit = True
            os.remove(STOP_FILE)
            print("\nStop file found — exiting gracefully.")
        return self.should_exit

    def stop(self):
        if self.listener:
            self.listener.stop()
