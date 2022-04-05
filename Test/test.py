from algorithms.GreedySearch import GreedySearch
from algorithms.MonteCarlo import MonteCarlo
from constants import game_constants
import numpy as np

grid = [[1, 4, 1, 4], [2, 2, 2, 0], [1, 3, 1, 6], [2, 5, 6, 7]]
# grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [16, 8, 2, 8]]
# print(game_util.is_down_possible(grid))
# print(game_util.is_up_possible(grid))
# print(game_util.is_left_possible(grid))
# print(game_util.is_right_possible(grid))
#
# print(game_util.get_possible_actions(grid))
# print(game_util.is_game_over(grid))
#
# print(game_util.move_up(grid))
# print(grid)
# print(game_util.get_monotonicity_heuristic(grid))
# print(game_util.get_empty_cell_heuristic(grid))
# print(game_util.get_smoothness_heuristic(grid))

# print(MonteCarlo(grid).make_move(grid, game_constants.LEFT))

print(np.count_nonzero(np.array(grid) == 0))
