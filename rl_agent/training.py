from r_learning import Q_agent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='Enter mode(Heuristic/Bucharin Heuristic')
args = parser.parse_args()

agent = Q_agent(n=2,mode = args.mode)

episodes = 10000

Q_agent.train_run(episodes, agent=agent, saving=True,mode = args.mode)
