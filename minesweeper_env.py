import numpy as np
import random
import pandas as pd
from IPython.display import display

class MinesweeperEnv:
    def __init__(self, n_rows=8, n_cols=8, n_mines=10,
                 r={'win': 5, 'lose': -1, 'progress': 0.3, 'no_progress': -0.3}):
        """
        Initialize Minesweeper environment for reinforcement learning

        Args:
            n_rows (int): Number of rows on the board
            n_cols (int): Number of columns on the board
            n_mines (int): Number of mines on the board
            r (dict): Reward values for different actions/outcomes
        """
        # Basic info about game setup
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_mines = n_mines
        self.n_cells = n_rows * n_cols
        self.r = r

        # Game state tracking
        self.grid = None  # Contains mine locations ('M' for mines, numbers for safe cells)
        self.board = None  # The fully revealed board (what the player would see if all cells were revealed)
        self.state = None  # Current state of the game (what's visible to the player)
        self.state_last = None  # Previous state for reward calculation

        # Track game outcomes
        self.won = 0
        self.lost = 0

        # Track number of clicks
        self.n_clicks = 0

        # Initialize the game
        self.reset()

    def create_grid(self):
        """Create a grid with mines randomly placed"""
        grid = np.zeros((self.n_rows, self.n_cols), dtype=object)
        mines_left = self.n_mines

        while mines_left > 0:
            i = random.randint(0, self.n_rows-1)
            j = random.randint(0, self.n_cols-1)
            if grid[i][j] != 'M':
                grid[i][j] = 'M'
                mines_left -= 1

        return grid

    def get_neighbours(self, tile):
        """Get the neighboring cells for a given tile"""
        x, y = tile[0], tile[1]
        neighbours = []

        for row in range(x-1, x+2):
            for col in range(y-1, y+2):
                if ((x != row or y != col) and
                    (0 <= row < self.n_rows) and
                    (0 <= col < self.n_cols)):
                    neighbours.append((row, col))

        return neighbours

    def create_board(self):
        """Create the fully revealed board with mine counts"""
        board = np.zeros((self.n_rows, self.n_cols), dtype=object)

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.grid[i][j] == 'M':
                    board[i][j] = 'M'
                else:
                    # Count mines in neighboring cells
                    mine_count = 0
                    for n_row, n_col in self.get_neighbours((i, j)):
                        if self.grid[n_row, n_col] == 'M':
                            mine_count += 1

                    if mine_count == 0:
                        board[i][j] = 'E'  # Empty cell
                    else:
                        board[i][j] = mine_count

        return board

    def init_game(self):
        """Initialize a new game"""
        self.grid = self.create_grid()
        self.board = self.create_board()

        # Initialize state - all cells unknown initially
        self.state = np.full((self.n_rows, self.n_cols), 'U', dtype=object)
        self.state_last = np.copy(self.state)

        # Reset click counter
        self.n_clicks = 0

    def reveal(self, row, col, checked):
        """Recursively reveal cells when an empty cell is clicked"""
        if checked[row, col] != 0:
            return

        checked[row, col] = 1
        if self.board[row, col] != 'M':
            # Reveal this cell in the state
            self.state[row, col] = self.board[row, col]

            # If empty, reveal neighbors recursively
            if self.board[row, col] == 'E':
                for n_row, n_col in self.get_neighbours((row, col)):
                    if not checked[n_row, n_col]:
                        self.reveal(n_row, n_col, checked)

    def action(self, a):
        """Take an action in the environment

        Args:
            a: Tuple (row, col) where to click

        Returns:
            dict with state, reward, and done flag
        """
        row, col = a

        # First move protection - ensure first click is never on a mine
        if self.n_clicks == 0 and self.board[row, col] == 'M':
            # Find a safe cell to relocate the first click
            safe_positions = []
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    if self.board[i, j] != 'M':
                        safe_positions.append((i, j))

            # Select a random safe position
            if safe_positions:
                row, col = random.choice(safe_positions)

        # Increment click counter
        self.n_clicks += 1

        # Game over if clicked on a mine (after first move protection check)
        if self.board[row, col] == 'M':
            self.lost += 1
            self.state[row, col] = 'M'  # Show the mine that was clicked
            return {"s": np.copy(self.state), "r": self.r['lose'], "d": True}

        # Store previous state for reward calculation
        self.state_last = np.copy(self.state)

        # Reveal cells based on the click
        self.reveal(row, col, np.zeros((self.n_rows, self.n_cols), dtype=int))

        # Check win condition: all non-mine cells revealed
        if np.sum(self.state == 'U') == self.n_mines:
            self.won += 1
            return {"s": np.copy(self.state), "r": self.r['win'], "d": True}

        # Calculate reward for this action
        reward = self.compute_reward(a)

        return {"s": np.copy(self.state), "r": reward, "d": False}

    def compute_reward(self, a):
        """Compute reward for the given action"""
        # Reward for revealing new cells (progress)
        if (np.sum(self.state_last == 'U') - np.sum(self.state == 'U')) > 0:
            reward = self.r['progress']
        else:
            reward = self.r['no_progress']

        return reward

    def get_state_representation(self, state):
        """Convert state to a format suitable for neural network input

        Two channel representation:
        - Channel 1: Number values (1-8)
        - Channel 2: Unknown cells (1 if unknown, 0 otherwise)
        """
        rows, cols = state.shape
        representation = np.zeros((rows, cols, 2))

        # Fill first channel with numerical values
        for i in range(rows):
            for j in range(cols):
                if isinstance(state[i, j], (int, np.integer)):
                    representation[i, j, 0] = state[i, j] / 8.0  # Normalize
                elif state[i, j] == 'E':
                    representation[i, j, 0] = 0

        # Fill second channel with unknown markers
        representation[:, :, 1] = (state == 'U').astype(float)

        return representation

    def num_state(self, state):
        """Alternative state representation from the provided code"""
        state_val = np.copy(state)
        state_val = np.reshape(state_val, (self.n_rows, self.n_cols, 1)).astype(object)
        state_val[state_val == 'U'] = -1
        state_val[state_val == 'M'] = -2
        state_val[state_val == 'E'] = 0
        # Convert to numeric and normalize
        state_val = state_val.astype(np.float32) / 8
        return state_val

    def color_state(self, value):
        """Apply colors to different cell values"""
        if value == 'U':
            return 'background-color: white'
        elif value == 'M':
            return 'background-color: red; color: black'
        elif value == 'E':
            return 'background-color: slategrey; color: white'
        elif isinstance(value, (int, np.integer)):
            colors = {
                1: 'blue',
                2: 'green',
                3: 'red',
                4: 'midnightblue',
                5: 'brown',
                6: 'aquamarine',
                7: 'black',
                8: 'silver'
            }
            color = colors.get(value, 'black')
            return f'color: {color}'
        else:
            return ''

    # Gym-style API
    def step(self, action):
        """Take a step in the environment (gym-style API)

        Args:
            action: Integer action (will be converted to row, col)

        Returns:
            observation, reward, done, info
        """
        # Convert flat action to 2D coordinates
        action_2d = np.unravel_index(action, (self.n_rows, self.n_cols))

        # Take action
        result = self.action(action_2d)

        return result["s"], result["r"], result["d"]

    def reset(self):
        """Reset the environment (gym-style API)"""
        self.init_game()
        return self.get_state_representation(self.state)

    def render(self, mode='human'):
        if mode == 'human':
            # Create DataFrame from current state
            df = pd.DataFrame(self.state)

            # Convert 'E' to 0 for display purposes
            df_display = df.copy()
            df_display = df_display.replace('E', 0)

            # Apply styling based on cell values
            styled_df = df_display.style.applymap(self.color_state)

            # Display the styled DataFrame
            display(styled_df)