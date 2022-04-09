import pygame
from pygame.locals import *
from r_learning import *
import sys
import argparse
from game_display import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='Enter mode(Snake Heuristic/N-Tuples Heuristic')
args = parser.parse_args()

# Loading the trained agent from best_agent.npy.
# Execute 100 episodes, replay the best episode.

agent = Q_agent.load_agent("best_agent.npy", args.mode)
est = agent.evaluate
results = Game.trial(estimator=est, num=100, mode = args.mode)

display = Display(matrix = results[0].row)