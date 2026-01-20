from pynput import keyboard


class ExitListener:
    def __init__(self, exit_key='%'):
        self.should_exit = False
        self.exit_key = exit_key
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        try:
            if key.char == self.exit_key:
                self.should_exit = True
                print("Exit key registered.")
        except AttributeError:
            pass

    def check_exit(self):
        return self.should_exit

    def stop(self):
        self.listener.stop()
