import numpy as np
import random

maze = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 2]
])

states = maze.size
actions = 4  # up, down, left, right
Q = np.zeros((states, actions))
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 1000

def state_to_pos(state):
    return divmod(state, maze.shape[1])

def pos_to_state(row, col):
    return row * maze.shape[1] + col

def get_reward(row, col):
    if maze[row, col] == 2:
        return 100
    elif maze[row, col] == 1:
        return -100
    return -1

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, actions - 1)
    return np.argmax(Q[state])

for _ in range(episodes):
    state = 0  # Start at (0,0)
    while True:
        action = choose_action(state)
        row, col = state_to_pos(state)
        if action == 0: row = max(0, row - 1)  # up
        elif action == 1: row = min(maze.shape[0] - 1, row + 1)  # down
        elif action == 2: col = max(0, col - 1)  # left
        elif action == 3: col = min(maze.shape[1] - 1, col + 1)  # right
        new_state = pos_to_state(row, col)
        reward = get_reward(row, col)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])
        state = new_state
        if reward == 100:
            break

print("Q-Table:")
print(Q)