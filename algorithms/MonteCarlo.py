from math import inf
import random

import numpy as np

from util import game_util


class MonteCarlo:

    def __init__(self, grid):
        self.grid = grid
        self.rollouts = 500

    def get_move(self):
        action_dict = {}
        for action in game_util.get_possible_actions(self.grid):
            move_value = 0
            child = self.make_move(self.grid, action)
            possible_actions = game_util.get_possible_actions(child)
            if len(possible_actions) != 0:
                for i in range(self.rollouts):
                    child_child = self.make_move(child, random.choice(possible_actions))
                    move_value += game_util.get_total_heuristic(child_child)
            action_dict[action] = move_value

        return max(action_dict, key=action_dict.get)

    def make_move(self, board, action):
        if game_util.action_possible[action](board):
            output, _, _ = game_util.action_functions[action](board)
            zero_index = random.choice(np.argwhere(np.array(output) == 0))
            output[zero_index[0]][zero_index[1]] = random.choices([4, 2], weights=[10, 90])[0]
            return output
        else:
            return board
