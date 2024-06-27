import numpy as np

v = np.zeros((5, 5, 5, 5))
v_prime = np.zeros((5, 5, 5, 5))
epsilon = 0.01
iter = 0

while True:
    max_diff = 0
    for agent_x in range(5):
        for agent_y in range(5):
            for target_x in range(5):
                for target_y in range(5):
                    if agent_x == target_x and agent_y == target_y:
                        continue
                    reward = 0
                    reward += v[max(0, agent_x-1), agent_y, target_x, target_y]
                    reward -= 1

                    reward += v[min(4, agent_x+1), agent_y, target_x, target_y]
                    reward -= 1

                    reward += v[agent_x, max(0, agent_y-1), target_x, target_y]
                    reward -= 1

                    reward += v[agent_x, min(4, agent_y+1), target_x, target_y]
                    reward -= 1

                    v_prime[agent_x, agent_y, target_x, target_y] = reward/4
                    max_diff = max(max_diff, abs(v[agent_x, agent_y, target_x, target_y] - v_prime[agent_x, agent_y, target_x, target_y]))
    iter += 1
    v = v_prime.copy()
    if max_diff <= epsilon:
        print(f"Policy evaluation is converged after {iter} iteration.")
        print(v[:,:,0,0])
        break