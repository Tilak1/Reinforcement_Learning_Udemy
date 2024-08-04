import gym 


class Lunar: 
    def __init__(self): 
        ngames= 1000 
        env = gym.make('LunarLander-v2')

        for i in range(ngames): 

            obs = env.reset()
            score = 0 
            done = False 

            while not done: 
                action = env.action_space.sample()
                obs,reward,done,info = env.step(action)
                score += reward 
            print('Episode number is',i,'Episode score %.1f'%score)


if __name__ == '__main__': 
    agent = Lunar()