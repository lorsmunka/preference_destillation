"""Which cells a formula reads, and the order to work cells out again after a change."""

from references import expand_range


def references_in(node):
    """Every cell name a formula tree reads, ranges expanded."""
    kind, value, children = node
    if kind == "reference":
        return {value}
    if kind == "range":
        return set(expand_range(value))
    return set().union(*(references_in(child) for child in children))


class DependencyGraph:
    def __init__(self):
        self.reads = {}   # cell -> the cells its formula reads

    def in_cycle(self, name):
        """True when a cell reads itself, directly or through other cells (a depth-first search)."""
        stack, seen = list(self.reads.get(name, ())), set()
        while stack:
            current = stack.pop()
            if current == name:
                return True
            if current not in seen:
                seen.add(current)
                stack.extend(self.reads.get(current, ()))
        return False

    def recalculation_order(self, name):
        """The cells that depend on a changed cell, each after the cells it reads (a topological sort)."""
        order = []
        self.visit(name, order, set())
        return order[::-1][1:]   # reversed post-order, without the changed cell itself

    def visit(self, name, order, visited):
        visited.add(name)
        for dependent in sorted(cell for cell, references in self.reads.items() if name in references):
            if dependent not in visited:
                self.visit(dependent, order, visited)
        order.append(name)
