import sys

from constants import game_constants
from game.game_display import Display

if __name__ == "__main__":
    args = sys.argv[1:]

    solver_index = args.index("-s")

    solver = args[solver_index + 1]


    if solver in game_constants.solver_map.keys():
        for i in range(15):
            p = Display(game_constants.solver_map[solver])
    else:
        print("No such solver")


