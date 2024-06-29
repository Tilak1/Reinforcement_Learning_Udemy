import numpy as np

class Agent:
    
    def __init__ (self,lr,gamma,nActions,nStates,epsStart,epsEnd,epsDec): 
    
        self.lr  = lr
        self.gamma = gamma
        self.nActions = nActions
        self.nStates = nStates
        self.epsilon = epsStart
        self.epsMin = epsEnd
        self.epsDec = epsDec
        self.Q = {}
        self.init_Q()
        
    def init_Q(self): 
        for state in range(self.nStates):
            for action in range(self.nActions): 
                self.Q[(state,action)]  = 0.0 
            
    def choose_action(self,state): 
        if np.random()<self.epsilon: 
            action = np.random.choice([i for i in range(self.nActions)])
        else: 
            actions = np.array([self.Q[(state,a)] for a in range(self.nActions)])
            action = np.argmx(actions) 
        return action 
    
    def epsilon_decrease(self): 
        self.epsilon = self.epsilon* self.epsDec if self.epsilon > self.epsMin else self.epsMin 
    
    def learn(self,state,action,reward,state_): 
        actions = np.array([self.Q[(state,a)] for a in range(self.nActions)])
        action_max = np.argmx(actions) 
        
        self.Q[(state,action)] += self.lr * (reward + self.gamma * self.Q[(state_,action_max)] - self.Q[(state,action)] )
        
        self.epsilon_decrease()
    
if __name__ == '__main__': 
     print("hellow")
     agent = Agent(lr=0.001, gamma=0.9, epsStart=1.0, epsEnd=0.01, epsDec=0.9999996, nActions=4, nStates=16)