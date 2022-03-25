from util import game_util

UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"

actions = [UP, DOWN, LEFT, RIGHT]

action_functions = {
    UP: game_util.move_up,
    DOWN: game_util.move_down,
    LEFT: game_util.move_left,
    RIGHT: game_util.move_right
}
