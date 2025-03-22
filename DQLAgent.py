import random
import os, sys
import numpy as np

ROOT = os.getcwd()
sys.path.insert(1, f'{os.path.dirname(ROOT)}')

import warnings
warnings.filterwarnings('ignore')

from collections import deque
from DQL import *
from minesweeper_env import *

# Environment settings
MEM_SIZE = 50_000 # number of moves to store in replay buffer
MEM_SIZE_MIN = 1_000 # min number of moves in replay buffer
TF_ENABLE_ONEDNN_OPTS=0

# Learning settings
BATCH_SIZE = 64
learn_rate = 0.01
LEARN_DECAY = 0.99975
LEARN_MIN = 0.001
DISCOUNT = 0.1 #gamma

# Exploration settings
epsilon = 0.95
EPSILON_DECAY = 0.975
EPSILON_MIN = 0.01

# DQN settings
CONV_UNITS = 64 # number of neurons in each conv layer
DENSE_UNITS = 512 # number of neurons in fully connected dense layer
UPDATE_TARGET_EVERY = 20

class DQLAgent(object):
    def __init__(self, env, conv_units = 64, dense_units = 256):
        self.env = env
        self.discount = DISCOUNT
        self.lr = learn_rate
        self.eps = epsilon
        in_dims = (*self.env.state.shape, 1)
        self.model = create_dqn(self.lr, in_dims, self.env.n_cells, conv_units, dense_units)

        self.target_model = create_dqn(
            self.lr, in_dims, self.env.n_cells, conv_units, dense_units)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.target_update_counter = 0

    def get_action(self, state):
        board = np.reshape(state, (1, self.env.n_cells))
        unsolved = [i for i, x in enumerate(board.flatten()) if x==-0.125]

        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.eps: # random move (explore)
            move = np.random.choice(unsolved)
            return move 
        
        moves = self.model.predict(np.reshape(state, (1, self.env.n_rows, self.env.n_cols, 1)), verbose = 0)
        moves[board!=-0.125] = -np.inf # set already clicked tiles to min value
        move = np.argmax(moves)

        return move

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, done):
        # Fix existing method
        if len(self.replay_memory) < MEM_SIZE_MIN:
            return

        batch = random.sample(self.replay_memory, BATCH_SIZE)
        current_states = np.array([transition[0] for transition in batch])
        current_qs_list = self.model.predict(current_states, verbose = 0)

        new_current_states = np.array([transition[3] for transition in batch])
        # Make sure to expand dimensions here too
        
        # Use the expanded tensor
        future_qs_list = self.target_model.predict(new_current_states, verbose = 0)

        X,y = [], []

        for i, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE,
                       shuffle=False, verbose=0 if done else None)

        # updating to determine if we want to update target_model yet
        if done:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # decay learn_rate
        self.lr = max(LEARN_MIN, self.lr*LEARN_DECAY)

        # decay epsilon
        self.eps = max(EPSILON_MIN, self.eps*EPSILON_DECAY)

if __name__ == "__main__":
    DQLAgent(MinesweeperEnv(8,8,10))