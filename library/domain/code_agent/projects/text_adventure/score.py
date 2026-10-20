"""The score: points for each new room, treasure and beaten monster, and the rank they earn."""

RANKS = [(150, "Hero"), (80, "Adventurer"), (30, "Explorer"), (0, "Beginner")]


class Score:
    def __init__(self, points=0, counted=()):
        self.points = points
        self.counted = set(counted)   # rooms, treasures and monsters already counted

    def award(self, thing, points):
        """Add points for a thing only the first time, so a dropped treasure cannot count twice."""
        if thing in self.counted:
            return 0
        self.counted.add(thing)
        self.points += points
        return points

    def rank(self):
        return next(title for needed, title in RANKS if self.points >= needed)
