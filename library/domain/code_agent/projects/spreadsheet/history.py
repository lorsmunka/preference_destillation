"""Undo: every step remembers what its cells held before."""

from config import MAX_UNDO


class History:
    def __init__(self):
        self.steps = []   # each step: a list of (cell name, text before), newest step last

    def record(self, changes):
        """Remember one step (a whole import is one); only the newest MAX_UNDO are kept."""
        if changes:
            self.steps.append(changes)
            del self.steps[:-MAX_UNDO]

    def undo(self, sheet):
        """Put back the cells of the newest step. The names of the cells that changed."""
        if not self.steps:
            return []
        changes = self.steps.pop()
        for name, text in reversed(changes):
            sheet.set_cell(name, text)
        return [name for name, _ in changes]
