import gym
from Q_Learning import Agent


if __name__ == '__main__': 
    env = gym.make('FrozenLake-v1')
    agent = Agent(lr=0.001, gamma=0.9, epsStart=1.0, epsEnd=0.01, epsDec=0.9999995, nActions=4, nStates=16)
     
    scores = []
    win_pct_list = []
    n_games = 500000
    
    for i in range(n_games): 
        done = False 
        observation = env.reset()
        score = 0 
        while not done: 
            action = agent.


