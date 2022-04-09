from r_learning import Q_agent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='Enter mode(Snake Heuristic/N-Tuples Heuristic')
parser.add_argument('--display', help='Enable Display During Training (y/n)')
parser.add_argument('--num', help='Number of Episodes')
args = parser.parse_args()

agent = Q_agent(n=2,mode = args.mode)

episodes = int(args.num)

Q_agent.train_run(episodes, agent=agent, saving=True,mode = args.mode, display_option = args.display)
#Q_agent.train_run(episodes, agent=agent, saving=True,mode = args.mode)
