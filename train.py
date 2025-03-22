import argparse, pickle
import os
import numpy as np
from tqdm import tqdm
from DQLAgent import *
from minesweeper_env import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# intake MinesweeperEnv parameters, beginner mode by default
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN to play Minesweeper')
    parser.add_argument('--width', type=int, default=9,
                        help='width of the board')
    parser.add_argument('--height', type=int, default=9,
                        help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10,
                        help='Number of mines on the board')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train on')
    parser.add_argument('--model_name', type=str, default='Minesweeper',
                        help='Name of model')

    return parser.parse_args()

params = parse_args()

MODEL_NAME = 'Our Model'

AGG_STATS_EVERY = 20 # calculate stats every 100 games for tensorboard
SAVE_MODEL_EVERY = 1000 # save model and replay every 10,000 episodes

def main():
    os.makedirs('replay', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    env = MinesweeperEnv(params.width, params.height, params.n_mines)
    agent = DQLAgent(env)

    progress_list, wins_list, ep_rewards = [], [], []
    n_clicks = 0

    for episode in tqdm(range(1, params.episodes+1), unit='episode'):
        # Reset environment
        env.reset()
        episode_reward = 0
        past_n_wins = env.won

        done = False
        while not done:
            # Get the current state in the format expected by DQLAgent
            current_state = env.num_state(env.state)  
            
            # Get action from agent
            action = agent.get_action(current_state)
            
            # Take step in environment
            new_state, reward, done = env.step(action)
            
            # Convert new state to the expected format
            new_state = env.num_state(new_state)
            
            episode_reward += reward

            # Store transition and train
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done)

            n_clicks += 1

        ep_rewards.append(episode_reward)
        
        # Calculate progress (cells revealed) for tracking
        progress = env.n_cells - env.n_mines - np.sum(env.state == 'U')
        progress_list.append(progress)

        if env.won > past_n_wins:
            wins_list.append(1)
        else:
            wins_list.append(0)

        if len(agent.replay_memory) < MEM_SIZE_MIN:
            continue

        if not episode % AGG_STATS_EVERY:
            med_progress = round(np.median(progress_list[-AGG_STATS_EVERY:]), 2)
            win_rate = round(np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY, 2)
            med_reward = round(np.median(ep_rewards[-AGG_STATS_EVERY:]), 2)

            print(f'Episode: {episode}, Median progress: {med_progress}, Median reward: {med_reward}, Win rate : {win_rate}')

        if not episode % SAVE_MODEL_EVERY:
            with open(f'replay/{MODEL_NAME}.pkl', 'wb') as output:
                pickle.dump(agent.replay_memory, output)

            agent.model.save(f'models/{MODEL_NAME}.h5')

if __name__ == "__main__":
    main()