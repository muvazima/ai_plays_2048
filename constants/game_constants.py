
UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"

actions = [UP, DOWN, LEFT, RIGHT]

monotonicity_weightage = 0.4
smoothness_weightage = 0.3
empty_cell_weightage = 0.2
max_value_weightage = 0.1

MANUAL = "Manual"
GREEDY = "Greedy"
EXPECTIMAX = "Expectimax"
MONTE_CARLO = "MonteCarlo"

solver_map = {"m": MANUAL, "g": GREEDY, "e": EXPECTIMAX, "mc": MONTE_CARLO}






