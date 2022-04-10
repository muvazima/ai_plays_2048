import pygame
from pygame.locals import *
from r_learning import *
import sys
import argparse
from game_display import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='Enter mode(Snake Heuristic/N-Tuples Heuristic')
parser.add_argument('--trials', help='Number of trials')

args = parser.parse_args()

# Loading the trained agent from best_agent.npy.
# Execute args.trials episodes, replay the best episode.

agent = Q_agent.load_agent("best_agent_trained_15000episodes.npy", args.mode)
est = agent.evaluate
results = Game.trial(estimator=est, num = int(args.trials), mode = args.mode)

# Displays the best game.
display = Display(matrix = results[0].row)