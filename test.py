import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tqdm import tqdm
from minesweeper_env import MinesweeperEnv
import os
import matplotlib.pyplot as plt

# Disable TensorFlow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    parser = argparse.ArgumentParser(description='Test a trained DQN on Minesweeper')
    parser.add_argument('--width', type=int, default=8,
                        help='width of the board')
    parser.add_argument('--height', type=int, default=8,
                        help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10,
                        help='Number of mines on the board')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to test on')
    parser.add_argument('--model_path', type=str, default='models/Our Model.h5',
                        help='Path to trained model')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the game board during testing')
    
    return parser.parse_args()

def get_action(model, state, env, epsilon=0.0):
    """Get action from the model with a small chance of random exploration"""
    # Reshape state to find unsolved tiles
    board = state.reshape(-1)
    
    # Find indices of unsolved cells (marked as -1 in num_state representation)
    unsolved = [i for i, x in enumerate(board) if x == -1]
    
    # Check if there are any unsolved cells left
    if not unsolved:
        # If no unsolved cells, return a random valid action
        return np.random.randint(0, env.n_cells)
    
    # Small chance of random action for exploration (can be set to 0 for pure exploitation)
    if np.random.random() < epsilon:
        return np.random.choice(unsolved)
    
    # Predict Q-values using the model
    moves = state.reshape(1, env.n_rows, env.n_cols, 1)
    moves = model.predict(moves, verbose=0)
    
    moves[board!=-0.125] = -np.inf # set already clicked tiles to min value
    move = np.argmax(moves)
    
    # Choose best action
    return move

def visualize_board(env):
    """Visualize the current state of the Minesweeper board"""
    plt.close('all')
    plt.figure(figsize=(8, 8))
    
    # Create a colormap for the board
    cmap = plt.cm.colors.ListedColormap(['white', 'grey', 'red', 'blue', 'green', 
                                        'orange', 'purple', 'maroon', 'turquoise', 'black'])
    
    # Create a numeric representation for visualization
    vis_board = np.zeros((env.n_rows, env.n_cols))
    
    for i in range(env.n_rows):
        for j in range(env.n_cols):
            if env.state[i, j] == 'U':
                vis_board[i, j] = 0 
            elif env.state[i, j] == 'M':
                vis_board[i, j] = 2
            elif env.state[i, j] == 'E':
                vis_board[i, j] = 1  # Empty - grey
            elif isinstance(env.state[i, j], (int, np.integer)):
                vis_board[i, j] = env.state[i, j] + 2 
    
    plt.imshow(vis_board, cmap=cmap, interpolation='nearest')
    
    # Add grid lines
    plt.grid(which='both', color='black', linestyle='-', linewidth=1)
    plt.xticks(np.arange(-0.5, env.n_cols, 1), [])
    plt.yticks(np.arange(-0.5, env.n_rows, 1), [])
    
    # Add text labels
    for i in range(env.n_rows):
        for j in range(env.n_cols):
            if isinstance(env.state[i, j], (int, np.integer)) and env.state[i, j] > 0:
                plt.text(j, i, str(env.state[i, j]), ha='center', va='center', color='white', fontweight='bold')
    
    plt.title('Minesweeper Board')
    plt.tight_layout()
    plt.pause(0.5)
    plt.clf() 

def main():
    # Parse command line arguments
    args = parse_args()
    args.model_path = "C:/Users/lahar/OneDrive/Documents/Minesweeper/models/Our Model.h5"
    
    # Create environment
    env = MinesweeperEnv(args.width, args.height, args.n_mines)
    
    # In test.py, modify the model loading section:
    try:
        # Define custom objects dictionary for loading
        custom_objects = {"mse": MeanSquaredError()}
        
        # Load model with custom objects
        model = load_model(args.model_path, custom_objects=custom_objects)
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Stats tracking
    wins = 0
    losses = 0
    avg_reveals = []
    
    # Run test episodes
    for episode in tqdm(range(args.episodes), desc="Testing"):
        # Reset environment
        env.reset()
        
        if args.visualize:
            plt.figure(figsize=(8, 8))
        
        # Track reveals for this episode
        reveals = 0
        
        # Play until game is done
        done = False
        while not done:
            # Get current state
            current_state = env.num_state(env.state)
            
            # Get action from model
            action = get_action(model, current_state, env, epsilon=0.00)
            
            # Take action
            new_state, reward, done = env.step(action)
            
            # Update reveals counter
            reveals += 1
            
            # Visualize if requested
            if args.visualize:
                visualize_board(env)
        
        # Update stats
        if env.won > 0:
            wins += 1
            avg_reveals.append(reveals)
        else:
            losses += 1
        
        if args.visualize:
            plt.close()
    
    # Calculate final stats
    win_rate = wins / args.episodes
    
    # Print results
    print("\n===== Test Results =====")
    print(f"Episodes played: {args.episodes}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win rate: {win_rate:.2%}")
    
if __name__ == "__main__":
    main()
