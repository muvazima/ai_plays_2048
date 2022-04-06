
UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT" 

actions = [UP, DOWN, LEFT, RIGHT]

monotonicity_weightage = 0.4
smoothness_weightage = 0.3
empty_cell_weightage = 0.1
max_value_weightage = 0.15

MANUAL = "Manual"
GREEDY = "Greedy"
MDQN = "MDQN"

solver_map = {"m": MANUAL, "g": GREEDY,"d" : MDQN}






