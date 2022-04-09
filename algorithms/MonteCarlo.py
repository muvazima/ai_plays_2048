import math
from math import inf
import random
from statistics import mean

import numpy as np

from constants.game_constants import actions
from util import game_util


class MonteCarlo:

    def __init__(self, grid):
        self.grid = grid
        self.rollouts = 100

    # def get_move(self):
    #     action_dict = {}
    #     for action in game_util.get_possible_actions(self.grid):
    #         move_value_merge = 0
    #         move_value_sum = 0
    #         move_value_max = 0
    #         child, merge_score = self.make_move(self.grid, action)
    #         possible_actions = game_util.get_possible_actions(child)
    #         move_value = [game_util.get_total_heuristic(child)]
    #         if len(possible_actions) != 0:
    #             for i in range(self.rollouts):
    #                 child, merge_score = self.make_move(child, random.choice(possible_actions))
    #                 move_value.append(game_util.get_total_heuristic(child))
    #                 # move_value_merge += merge_score
    #                 # move_value_sum += np.sum(child_child)
    #                 # move_value_max += np.max(child)
    #         action_dict[action] = mean(move_value)
    #
    #     return max(action_dict, key=action_dict.get)

    # def get_move(self):
    #     action_dict = {}
    #     for action in game_util.get_possible_actions(self.grid):
    #         move_value_merge = 0
    #         move_value_sum = 0
    #         move_value_max = 0
    #         child, merge_score = self.make_move(self.grid, action)
    #         possible_actions = game_util.get_possible_actions(child)
    #         move_value = []
    #
    #         # print(possible_actions)
    #         if len(possible_actions) != 0:
    #             for i in range(self.rollouts):
    #                 child_child = child.copy()
    #                 s = 0
    #                 rollout_score = 0
    #                 while not game_util.is_game_over(child_child):
    #                     s += 1
    #                     # print(s, end=" "),
    #                     child_child, merge_score = self.make_move(
    #                                                     child_child,
    #                                                     random.choice(game_util.get_possible_actions(child_child)))
    #                     rollout_score = np.max(child_child)
    #                     # rollout_score += merge_score
    #                     # print(child_child)
    #
    #                 move_value.append(rollout_score)
    #                 # move_value_merge += merge_score
    #                 # move_value_sum += np.sum(child_child)
    #                 # move_value_max += np.max(child)
    #         action_dict[action] = mean(move_value)
    #     print(action_dict)
    #     return max(action_dict, key=action_dict.get)
    #
    def make_move(self, board, action):
        if game_util.action_possible[action](board):
            output, _, merge_score = game_util.action_functions[action](board)
            zero_index = random.choice(np.argwhere(np.array(output) == 0))
            output[zero_index[0]][zero_index[1]] = random.choices([4, 2], weights=[10, 90])[0]
            return output, merge_score
        else:
            return board, 0

    def get_move(self):
        Q = {}
        N = {}
        T = set()
        depth = 4
        for i in range(self.rollouts):
            self.simulate(self.grid, Q, T, N, depth)

        possible_actions = game_util.get_possible_actions(self.grid)
        max_index = np.argmax(np.array([Q[self.flatten(self.grid), action] for action in possible_actions]))
        best_action = possible_actions[max_index]
        return best_action

    def flatten(self, t):
        a = [item for sublist in t for item in sublist]
        return "".join(map(str, a))

    def simulate(self, board, Q, T, N, depth):
        # print(depth, end=" ")
        if depth == 0 or game_util.is_game_over(board):
            return 0

        possible_actions = game_util.get_possible_actions(board)
        if self.flatten(board) not in T:
            for action in possible_actions:
                _, _, Q[self.flatten(board), action] = game_util.action_functions[action](board)
                N[self.flatten(board), action] = 1

            T.add(self.flatten(board))
            return self.rollout(board, depth)
        else:
            best_action = self.get_best_action(board, Q, N)
            child_board, score = self.make_move(board, best_action)
            q_score = score + self.simulate(child_board, Q, T, N, depth - 1)
            N[self.flatten(board), best_action] += 1
            Q[self.flatten(board), best_action] += (q_score - Q[self.flatten(board), best_action]) \
                                                   / N[self.flatten(board), best_action]

            return q_score

    def get_best_action(self, board, Q, N):
        possible_actions = game_util.get_possible_actions(board)
        # for/  / action in possible_actions:
        Ns = sum([N[self.flatten(board), action] for action in possible_actions])

        action_value = {}
        for action in possible_actions:
            action_value[action] = Q[self.flatten(board), action] + \
                                   100 * math.sqrt(math.log(Ns) / N[self.flatten(board), action])

        return max(action_value, key=action_value.get)

    def rollout(self, board, depth):
        if depth == 0 or game_util.is_game_over(board):
            return 0
        possible_actions = game_util.get_possible_actions(board)
        action_value = {}
        for action in possible_actions:
            child_board, _, merge_score = game_util.action_functions[action](board)
            action_value[action] = merge_score

        best_action = max(action_value, key=action_value.get)
        next_board, _ = self.make_move(board, best_action)
        return action_value[best_action] + 0.75 * self.rollout(next_board, depth - 1)
