import time
from tkinter import Frame, Label, CENTER

import game_functions
from algorithms.Expectimax import Expectimax
from algorithms.MonteCarlo import MonteCarlo
from constants import game_constants
from util import game_util

from algorithms.GreedySearch import GreedySearch

EDGE_LENGTH = 400
CELL_COUNT = 4
CELL_PAD = 10

UP_KEY = "'w'"
DOWN_KEY = "'s'"
LEFT_KEY = "'a'"
RIGHT_KEY = "'d'"
AI_KEY = "'q'"
AI_PLAY_KEY = "'p'"
GREEDY_KEY = "'g'"

LABEL_FONT = ("Verdana", 40, "bold")

GAME_COLOR = "#a6bdbb"

EMPTY_COLOR = "#8eaba8"

TILE_COLORS = {2: "#daeddf", 4: "#9ae3ae", 8: "#6ce68d", 16: "#42ed71",
               32: "#17e650", 64: "#17c246", 128: "#149938",
               256: "#107d2e", 512: "#0e6325", 1024: "#0b4a1c",
               2048: "#031f0a", 4096: "#000000", 8192: "#000000", }

LABEL_COLORS = {2: "#011c08", 4: "#011c08", 8: "#011c08", 16: "#011c08",
                32: "#011c08", 64: "#f2f2f0", 128: "#f2f2f0",
                256: "#f2f2f0", 512: "#f2f2f0", 1024: "#f2f2f0",
                2048: "#f2f2f0", 4096: "#f2f2f0", 8192: "#f2f2f0", }


class Display(Frame):
    def __init__(self, solver):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_press)

        self.commands = {
            UP_KEY: game_functions.move_up,
            DOWN_KEY: game_functions.move_down,
            LEFT_KEY: game_functions.move_left,
            RIGHT_KEY: game_functions.move_right
            # GREEDY_KEY = GreedySearch()
        }

        self.grid_cells = []
        self.build_grid()
        self.init_matrix()
        # self.draw_grid_cells()
        self.update(solver)
        # self.mainloop()

    def update(self, solver):
        i = 0
        move_made = None
        # flag1024 = False
        # flag2048 = False
        # flag4096 = False
        # flag8192 = False
        # stats_dict = {'execution_time': 0, 'execution_time_iter': 0, 'num_iters': 0, "max_num": 0,
        #               'time_1024': 0, 'time_2048': 0, 'time_4096': 0, 'time_8192': 0, 'depth': 0, 'roll_out': 0}

        start_time = time.time()
        while not game_util.is_game_over(self.matrix) and solver != game_constants.MANUAL:
            if solver == game_constants.GREEDY:
                move_made = GreedySearch(self.matrix).get_move()
            if solver == game_constants.EXPECTIMAX:
                move_made = Expectimax(self.matrix).get_move()
            if solver == game_constants.MONTE_CARLO:
                move_made = MonteCarlo(self.matrix).get_move()
            if move_made:
                self.matrix, _, _ = game_util.action_functions[move_made](self.matrix)
                self.matrix = game_functions.add_new_tile(self.matrix)
                self.draw_grid_cells()
                move_made = False
                print(self.matrix)

        self.mainloop()

    def build_grid(self):
        background = Frame(self, bg=GAME_COLOR,
                           width=EDGE_LENGTH, height=EDGE_LENGTH)
        background.grid()

        for row in range(CELL_COUNT):
            grid_row = []
            for col in range(CELL_COUNT):
                cell = Frame(background, bg=EMPTY_COLOR,
                             width=EDGE_LENGTH / CELL_COUNT,
                             height=EDGE_LENGTH / CELL_COUNT)
                cell.grid(row=row, column=col, padx=CELL_PAD,
                          pady=CELL_PAD)
                t = Label(master=cell, text="",
                          bg=EMPTY_COLOR,
                          justify=CENTER, font=LABEL_FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def init_matrix(self):
        self.matrix = game_functions.initialize_game()

    def draw_grid_cells(self):
        for row in range(CELL_COUNT):
            for col in range(CELL_COUNT):
                tile_value = self.matrix[row][col]
                if not tile_value:
                    self.grid_cells[row][col].configure(
                        text="", bg=EMPTY_COLOR)
                else:
                    self.grid_cells[row][col].configure(text=str(
                        tile_value), bg=TILE_COLORS[tile_value],
                        fg=LABEL_COLORS[tile_value])
        self.update_idletasks()

    def key_press(self, event):
        print(self.matrix)
        valid_game = True
        key = repr(event.char)
        if key == GREEDY_KEY:
            move_made = GreedySearch(self.matrix).get_move()
            self.matrix, _, _ = game_util.action_functions[move_made](self.matrix)
            if move_made:
                self.matrix = game_functions.add_new_tile(self.matrix)
                self.draw_grid_cells()
                move_made = False
        # if key == AI_PLAY_KEY:
        #     move_count = 0
        #     while valid_game:
        #         self.matrix, valid_game = game_ai.ai_move(self.matrix, 40, 30)
        #         if valid_game:
        #             self.matrix = game_functions.add_new_tile(self.matrix)
        #             self.draw_grid_cells()
        #         move_count += 1
        # if key == AI_KEY:
        #     self.matrix, move_made = game_ai.ai_move(self.matrix, 20, 30)
        #     if move_made:
        #         self.matrix = game_functions.add_new_tile(self.matrix)
        #         self.draw_grid_cells()
        #         move_made = False

        elif key in self.commands:
            self.matrix, move_made, _ = self.commands[repr(event.char)](self.matrix)
            if move_made:
                self.matrix = game_functions.add_new_tile(self.matrix)
                self.draw_grid_cells()
                move_made = False
