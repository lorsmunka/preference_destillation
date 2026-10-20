"""Settings of the fitness tracker: the data file, the goals, the units and the reminders."""

DATA_FILE = "fitness.json"
UNIT = "km"                  # "km" or "mi"; the u menu option switches it
KM_PER_MILE = 1.609344
STRIDE_KM = 0.00075          # the length of one step

DAILY_STEP_GOAL = 8000
WEEKLY_GOALS = {"minutes": 150, "workouts": 3, "km": 15}

REFERENCE_WEIGHT_KG = 70     # the calorie table is for a person of this weight
MOVING_AVERAGE_DAYS = 7
STEADY_CHANGE_KG = 0.2       # a smaller change of the average counts as steady
NUDGE_AFTER_DAYS = 2
