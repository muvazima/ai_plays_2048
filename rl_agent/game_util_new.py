import math

#from constants.game_constants import *
import game_functions
import numpy as np
from game_constants import *
#from constants import game_constants


def get_cell_value(grid, index):
    return grid[index[0]][index[1]]


def get_cell_log_value(grid, index):
    value = get_cell_value(grid, index)

    return get_log(value)


def get_log(num):
    if num == 0:
        return 0
    return math.log(num, 2)


def is_left_possible(grid):
    for row in grid:
        for cell_index in range(len(row) - 1):
            if (row[cell_index] == 0 and row[cell_index + 1] != 0) or row[cell_index] == row[cell_index + 1]:
                return True
    return False


def is_right_possible(grid):
    for row in grid:
        for cell_index in range(len(row) - 1, 0, -1):
            if (row[cell_index] == 0 and row[cell_index - 1] != 0) or row[cell_index] == row[cell_index - 1]:
                return True
    return False


def is_up_possible(grid):
    transposed_grid = np.array(grid).T
    for row in transposed_grid:
        for cell_index in range(len(row) - 1):
            if (row[cell_index] == 0 and row[cell_index + 1] != 0) or row[cell_index] == row[cell_index + 1]:
                return True
    return False


def is_down_possible(grid):
    transposed_grid = np.array(grid).T
    for row in transposed_grid:
        for cell_index in range(len(row) - 1, 0, -1):
            if (row[cell_index] == 0 and row[cell_index - 1] != 0) or row[cell_index] == row[cell_index - 1]:
                return True
    return False


def move_left(grid):
    return game_functions.move_left(grid)


def move_right(grid):
    return game_functions.move_right(grid)


def move_up(grid):
    return game_functions.move_up(grid)


def move_down(grid):
    return game_functions.move_down(grid)


action_functions = {
    UP: move_up,
    DOWN: move_down,
    LEFT: move_left,
    RIGHT: move_right
}


def get_possible_actions(grid):
    actions = []
    for action, func in zip(actions,
                            [is_up_possible, is_down_possible, is_left_possible, is_right_possible]):
        if func(grid):
            actions.append(action)
    return actions


def is_game_over(grid):
    return not bool(get_possible_actions(grid))


def get_total_heuristic(grid):
    return sum([monotonicity_weightage * get_monotonicity_heuristic(grid),
                smoothness_weightage * get_smoothness_heuristic(grid),
                empty_cell_weightage * get_empty_cell_heuristic(grid),
                max_value_weightage * get_max_value_heuristic(grid)])
    #return snake_heuristic(grid)


def get_monotonicity_heuristic(grid):
    left_diff, right_diff, up_diff, down_diff = 0, 0, 0, 0
    grid_transpose = np.array(grid).T

    for i in range(len(grid)):
        for j in range(len(grid[0]) - 1):
            diff_x = get_cell_log_value(grid, [i, j]) - get_cell_log_value(grid, [i, j + 1])
            if diff_x >= 0:
                # left_diff += diff_x
                left_diff += 1
            else:
                # right_diff += abs(diff_x)
                right_diff += 1

            diff_y = get_cell_log_value(grid_transpose, [i, j]) - get_cell_log_value(grid_transpose, [i, j + 1])
            if diff_y >= 0:
                # up_diff += diff_y
                up_diff += 1
            else:
                # down_diff += abs(diff_y)
                down_diff += 1

    total = 2*(max(left_diff, right_diff) + max(up_diff, down_diff))
    return total


def get_smoothness_heuristic(grid):
    total = 0

    for row in grid:
        np_row = np.array(row)
        non_zero_elements = np_row[np.where(np_row != 0)]
        for i in range(len(non_zero_elements) - 1):
            total -= abs(get_log(non_zero_elements[i]) - get_log(non_zero_elements[i + 1]))

    grid_transpose = np.array(grid).T
    for row in grid_transpose:
        np_row = np.array(row)
        non_zero_elements = np_row[np.where(np_row != 0)]
        for i in range(len(non_zero_elements) - 1):
            total -= abs(get_log(non_zero_elements[i]) - get_log(non_zero_elements[i + 1]))
    return total


def get_empty_cell_heuristic(grid):
    return np.count_nonzero(np.array(grid) == 0)


def get_max_value_heuristic(grid):
    return math.log(np.max(grid), 2)

def snake_heuristic(grid):
    return np.sum(np.array([[2**15, 2**14, 2**13, 2**12],
                  [2**8, 2**9, 2**10, 2**11],
                  [2**7, 2**6, 2**5, 2**4],
                  [2**0, 2**1, 2**2, 2**3]]) * np.array(grid))

