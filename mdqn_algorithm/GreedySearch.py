from util import game_util


class GreedySearch:

    def __init__(self, grid):
        self.name = ""
        self.grid = grid

    def get_move(self):
        action_dict = {}
        for action in game_util.get_possible_actions(self.grid):
            action_dict[action] = self.evaluate(action)
        return max(action_dict, key=action_dict.get)

    def evaluate(self, action):
        temp_grid, _, _ = game_util.action_functions[action](self.grid)
        value = game_util.get_total_heuristic(temp_grid)
        return value
