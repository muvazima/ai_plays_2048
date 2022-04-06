from r_learning import Q_agent

agent = Q_agent(n=2)

episodes = 500

Q_agent.train_run(episodes, agent=agent, saving=False)