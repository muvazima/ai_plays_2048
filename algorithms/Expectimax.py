from math import inf

import numpy as np

from util import game_util


class Expectimax:

    def __init__(self, grid):
        self.grid = grid

    def get_move(self):
        depth = 2
        action_dict = {}
        for action in game_util.get_possible_actions(self.grid):
            child, _, _ = game_util.action_functions[action](self.grid)
            action_dict[action] = self.evaluate(child, depth, chance=True)
        return max(action_dict, key=action_dict.get)

    def evaluate(self, board, depth, chance):
        if depth == 0 or game_util.is_game_over(board):
            return game_util.get_total_heuristic(board)

        alpha = 0
        if not chance:
            alpha = -inf
            for action in game_util.get_possible_actions(board):
                child, _, _ = game_util.action_functions[action](board)
                alpha = max(alpha, self.evaluate(child, depth - 1, chance=True))
        else:
            zero_index_list = np.argwhere(np.array(board) == 0)

            for index in zero_index_list:

                child_with_2 = board.copy()
                child_with_2[index[0]][index[1]] = 2
                child_with_4 = board.copy()
                child_with_4[index[0]][index[1]] = 4
                alpha += (0.9 * self.evaluate(child_with_2, depth=depth - 1, chance=False) \
                         + 0.1 * self.evaluate(child_with_4, depth=depth - 1, chance=False))/len(zero_index_list)

        return alpha
