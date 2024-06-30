import gym
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim

from utils import plot_learning_curve

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        return actions

class Agent():
    def __init__(self, input_dims, lr, nActions, gamma=0.99, epsilon=1.0, epsmin=0.01, epsdec=1e-5):
        self.lr = lr
        self.gamma = gamma
        self.input_dims = input_dims
        self.nActions = nActions
        self.epsilon = epsilon
        self.epsMin = epsmin
        self.epsDec = epsdec
        self.actionspace = [i for i in range(self.nActions)]
        self.Q = LinearDeepQNetwork(self.lr, self.nActions, self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.actionspace)
        return action
        
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsDec if self.epsilon > self.epsMin else self.epsMin

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        
        # Ensure state and state_ are sequences of correct length
        print(f"Original state shape: {state.shape}")
        print(f"Original state_ shape: {state_.shape}")

        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        if state_.ndim == 1:
            state_ = np.expand_dims(state_, axis=0)

        print(f"Reshaped state shape: {state.shape}")
        print(f"Reshaped state_ shape: {state_.shape}")

        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor([action]).to(self.Q.device)
        rewards = T.tensor([reward]).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[0][actions]
        q_next = self.Q.forward(states_).max()
        q_target = rewards + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []
    
    agent = Agent(input_dims=env.observation_space.shape, lr=0.0001, nActions=env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, _, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_

        scores.append(score)
        eps_history.append(agent.epsilon)
        agent.decrement_epsilon()

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'episode {i} score {score:.1f} avg score {avg_score:.1f} epsilon {agent.epsilon:.2f}')

    filename = 'cartpole_dqn.png'
    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)
