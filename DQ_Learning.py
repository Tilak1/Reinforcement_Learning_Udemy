#import __main__
import gym 
import numpy as np 
import matplotlib.pyplot as plt 

import torch.nn as nn
import torch as T 
import torch.nn.functional as F 
import torch.optim as optim 

from utils import plot_learning_curve

class LinearDeepQNetwork(nn.Module): #inheriting nn.module helps with self.parameters() from the optimizer 
    def __init__(self,lr,n_actions,input_dims): 
        super(LinearDeepQNetwork,self).__init__() # ineriting nn.module's 
        
        
        # n_actions dimension because we need to have Q values for every state's action pair 
        #So the output should be of n_actions dimension
        
        self.fc1 = nn.Linear(input_dims[0],128)
        self.fc2 = nn.Linear(128,n_actions)
        
        # Opttimizer is the grad descent and here Adam / SGD are some of the GD processess you can choose from 
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        
        self.loss = nn.MSELoss()
        self.device = T.device('cud:0' if T.cuda.is_available() else 'cpu')
        
        #sending entire network to device 
        self.to(self.device)
        
    """ def forward (self,state): # taking state as inpuyt and not data here 
        layer1 = F.relu(self.fc1(state))
        #layer2 = F.relu(self.fc2(layer1))
        actions = self.fc2 (layer1)
        
        return actions      # not returing the final layer, but actions
     """
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        return actions

    
    # def learn (self,data,labels):
    #     self.optimizer.zero_grad()
    #     data = T.tensor(data).to(self.device)
    #     labels = T.tensor(labels).to(self.device)
        
    #     predictions = self.forward (data) 
    #     cost = self.loss(predictions,labels)
        
    #     cost.backward()
    #     self.optimizer.step()

class Agent(): 
    def __init__(self,input_dims,lr,nActions,gamma=0.99,epsilon = 1.0, epsmin = 0.01,epsdec=1e-5): 
        
        self.lr  = lr
        self.gamma = gamma
        self.input_dims = input_dims
        self.nActions = nActions
        #self.nStates = nStates
        self.epsilon = epsilon
        self.epsMin = epsmin
        self.epsDec = epsdec
        # defining an action space - list of integers from 0 to nActions
        self.actionspace = [i for i in range(self.nActions)]
        
        self.Q = LinearDeepQNetwork(self.lr,self.nActions,self.input_dims)
        #creating an object of the above class 
        # Q estimate is one aspect of the agent 
        
        #self.init_Q()

    #def init_Q(): 
        
    def choose_action(self,observation):
        # generate a randome value and based on that we will choose the epsilon action 
        #epsilon : greedy -> use the tensor.item()
        # zero the grad
        
        # if np.random.random() > self.epsilon: 
        #     # conv observation is a pytorch tensor - that it sent to the GPU device 
        #     # as device is a property of the LinearDeepQNetwork class. So using self.Q -> device 
        #     # making sure tenosr is of float - so that every thing matches up 
        #     state = T.tensor(observation,dtype = T.float).to(self.Q.device)
        
        #     actions = self.Q.forward(state)
        #     action = T.argmax(actions).item() #use the tensor.item() to get numpy item out of it 
        
        state = observation[0] if isinstance(observation, tuple) else observation
        state = T.tensor(state, dtype=T.float).to(self.Q.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if np.random.random() > self.epsilon:
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.actionspace)
        return action
        
    def decrement_gamma(self):
        self.epsilon = self.epsilon  - self.epsDec if self.epsilon > self.epsMin else self.epsMin
        

    def learn (self,state,action,reward,state_): 
        
        # Grad descent flow: 
        # zero grad 
        # loss calculate 
        # Backward prop 
        # step the grad at last 
        
        state_array = state[0] if isinstance(state, tuple) else state
        state__array = state_[0] if isinstance(state_, tuple) else state_

        states = T.tensor(state_array, dtype=T.float).to(self.Q.device)
        actions = T.tensor([action],dtype=T.long).to(self.Q.device)
        rewards = T.tensor([reward]).to(self.Q.device)
        states_ = T.tensor(state__array, dtype=T.float).to(self.Q.device)

        # Ensure states and states_ are 2D
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if states_.dim() == 1:
            states_ = states_.unsqueeze(0)

        q_pred = self.Q.forward(states)
        q_next = self.Q.forward(states_)

        max_next_q = q_next.max(dim=1)[0]
        q_target = rewards + self.gamma * max_next_q

        q_target = q_target.unsqueeze(1)
        loss = self.Q.loss(q_target, q_pred.gather(1, actions.unsqueeze(1))).to(self.Q.device)
        
        self.Q.optimizer.zero_grad()
        loss.backward()
        self.Q.optimizer.step()
        
if __name__ == "__main__":         
   
    env = gym.make('CartPole-v1')
    n_games = 10000 
    scores = []
    eps_hstory = []
    
    agent = Agent(input_dims=[env.observation_space.shape[0]],lr=0.0001,nActions=env.action_space.n) 

    for i in range(n_games): 
        score = 0 
        done = False 
        obs,_ = env.reset()
        
        while not done:

            action = agent.choose_action(obs) # choosing an action acc to the epsilon greedy selection strategy 
            obs_,reward, done, _, info =  env.step(action)
            score += reward 
            agent.learn(obs,int(action),float(reward),obs_) # learn from the state action reward & popiulte the Q values 
            obs = obs_ # setting old state to the new state 
            
        scores.append(score)
        eps_hstory.append(agent.epsilon)
        
        
        if i % 100 == 0: # mean scores over the last 100 games 
            avg_score = np.mean(scores[-100:]) 
            
            print('episode', i , 'score %.1f avg score %.1f epsilon %.2f' %(score,avg_score,agent.epsilon))
     
    filename = 'cartpole_nive_dqn.png' # why file name ? 
    x = [i+1 for i in range(n_games)] # cretaing a list of no_of_games to plot in x axis         
    plot_learning_curve(x,scores,eps_hstory,filename)