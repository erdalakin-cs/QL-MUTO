import numpy as np
import pandas as pd

import matplotlib
import matplotlib.style


class QLearningTable:
    def __init__(self, actions, lr = 0.15, gamma = 0.5, eps = 1.0):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.q_table = pd.DataFrame(columns=self.actions,dtype=np.float64)

    def choose_action(self, obs):

        self.check_state_exist(obs)

        if np.random.rand() < self.eps:
            action = np.random.choice(self.actions)

        else:
            state_action = self.q_table.loc[obs,:]  
            action = np.argmax(state_action) 

        return action 

    def learn(self, s, a, r,s_):
        self.check_state_exist(s_)

        q_predict = self.q_table.loc[s, a]
        #if s_ != 'terminal':
        q_target = r + self.gamma * self.q_table.loc[s_, :].max() #np.argmax(self.q_table.loc[s_,:])#self.q_table.loc[s_, :].max()  # next state is not terminal
       # else:
       #     q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )