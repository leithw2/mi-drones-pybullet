import numpy as np
import random

# --- Parámetros ---
grid_size = 60
actions = [0,1,2,3]  # up, down, left, right
alpha = 0.1
gamma = 0.90
epsilon = 0.2
episodes = 10

# --- Q-table ---
Q = np.zeros((grid_size, grid_size, len(actions)))

# --- Entorno ---
goal = (57,47)
obstacles = [(25,25), (30,32), (10,10), (11,10), (12,10), (13,10),
             (25+10,25), (30+10,32), (10+10,10), (11+10,10), (12+10,10), (13+10,10),
             (25,25+10), (30,32+10), (10,10+10), (11,10+10), (12,10+10), (13,10+10),
             (14,10), (14,9), (14,8), (14,7), (14,6), (13,6), (13,6)]

def step(state, action):
    x, y = state

    if action == 0: x -= 1
    if action == 1: x += 1
    if action == 2: y -= 1
    if action == 3: y += 1

    # límites
    x = max(0, min(grid_size-1, x))
    y = max(0, min(grid_size-1, y))

    next_state = (x,y)

    # recompensas
    if next_state == goal:
        return next_state, 200, True
    elif next_state in obstacles:
        return next_state, -20, True
    else:
        return next_state, -.2, False

# --- Política epsilon-greedy ---
def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        x,y = state
        return np.argmax(Q[x,y])
max_reward = None
# --- Entrenamiento ---
for ep in range(episodes):
    state = (np.random.randint(0, 10), np.random.randint(0, 10))
    reward_total = 0
    done = False
    dist = 0
    hold_reward = []
    
    while not done:
        x,y = state
        action = choose_action(state)

        next_state, reward, done = step(state, action)

        nx, ny = next_state

        # Q-learning update
        Q[x,y,action] += alpha * (
            reward + gamma * np.max(Q[nx,ny]) - Q[x,y,action]
        )
        reward_total += reward
        dist += 1 
        state = next_state
    max_reward = reward_total if max_reward is None or max_reward < reward_total else max_reward
    hold_reward.append(reward_total)
    average_reward = np.mean(hold_reward)
    if (ep + 1) % 1000 == 0: # Mostrar progreso cada 1000 episodios
        print(f"Episode {ep+1}, Max Reward: {max_reward}, Reward: {reward_total}, Steps: {dist}, Average Reward: {average_reward}")

# --- Mostrar política aprendida ---
policy = np.full((grid_size, grid_size), ' ')

symbols = ['↑','↓','←','→']

for i in range(grid_size):
    for j in range(grid_size):
        if (i,j) == goal:
            policy[i,j] = 'G'
        elif (i,j) in obstacles:
            policy[i,j] = 'X'
        else:
            best_action = np.argmax(Q[i,j])
            policy[i,j] = symbols[best_action]

print("Política aprendida:\n")
for row in policy:
    print(' '.join(row))