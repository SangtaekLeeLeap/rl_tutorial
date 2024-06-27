import numpy as np

policy = np.empty(shape=(5, 5, 5, 5, 4))
policy.fill(0.25)
v = np.zeros((5, 5, 5, 5))
v_prime = np.zeros((5, 5, 5, 5))
iter = 0

while True:
    for agent_x in range(5):
        for agent_y in range(5):
            for target_x in range(5):
                for target_y in range(5):
                    if agent_x == target_x and agent_y == target_y:
                        continue
                    reward = np.zeros(4)
                    reward[0] += v[max(0, agent_x-1), agent_y, target_x, target_y]
                    reward[0] -= 1
                    reward[0] *= policy[agent_x, agent_y, target_x, target_y, 0]

                    reward[1] += v[min(4, agent_x+1), agent_y, target_x, target_y]
                    reward[1] -= 1
                    reward[1] *= policy[agent_x, agent_y, target_x, target_y, 1]

                    reward[2] += v[agent_x, max(0, agent_y-1), target_x, target_y]
                    reward[2] -= 1
                    reward[2] *= policy[agent_x, agent_y, target_x, target_y, 2]

                    reward[3] += v[agent_x, min(4, agent_y+1), target_x, target_y]
                    reward[3] -= 1
                    reward[3] *= policy[agent_x, agent_y, target_x, target_y, 3]

                    v_prime[agent_x, agent_y, target_x, target_y] = np.sum(reward)

    v = v_prime.copy()
    print(f"Policy evaluation is done.")
    print(v[:,:,0,0])

    policy_old = policy.copy()
    for agent_x in range(5):
        for agent_y in range(5):
            for target_x in range(5):
                for target_y in range(5):
                    if agent_x == target_x and agent_y == target_y:
                        continue
                    max_reward = max(v[max(0, agent_x-1), agent_y, target_x, target_y], v[min(4, agent_x+1), agent_y, target_x, target_y], v[agent_x, max(0, agent_y-1), target_x, target_y], v[agent_x, min(4, agent_y+1), target_x, target_y])
                    
                    count = 0
                    if (v[max(0, agent_x-1), agent_y, target_x, target_y] == max_reward):
                        count += 1
                        policy[agent_x, agent_y, target_x, target_y, 0] = 1
                    else:
                        policy[agent_x, agent_y, target_x, target_y, 0] = 0

                    if (v[min(4, agent_x+1), agent_y, target_x, target_y] == max_reward):
                        count += 1
                        policy[agent_x, agent_y, target_x, target_y, 1] = 1
                    else:
                        policy[agent_x, agent_y, target_x, target_y, 1] = 0
                    
                    if (v[agent_x, max(0, agent_y-1), target_x, target_y] == max_reward):
                        count += 1
                        policy[agent_x, agent_y, target_x, target_y, 2] = 1
                    else:
                        policy[agent_x, agent_y, target_x, target_y, 2] = 0

                    if (v[agent_x, min(4, agent_y+1), target_x, target_y] == max_reward):
                        count += 1
                        policy[agent_x, agent_y, target_x, target_y, 3] = 1
                    else:
                        policy[agent_x, agent_y, target_x, target_y, 3] = 0

                    policy[agent_x, agent_y, target_x, target_y, 0] /= count
                    policy[agent_x, agent_y, target_x, target_y, 1] /= count
                    policy[agent_x, agent_y, target_x, target_y, 2] /= count
                    policy[agent_x, agent_y, target_x, target_y, 3] /= count
    iter += 1
    print(f"Policy improvement is done.")
    print(policy[:,:,0,0,:])
    if np.array_equal(policy_old, policy):
        print(f"Policy iteration is converged after {iter} iteration")
        print(v[:,:,0,0])
        print(policy[:,:,0,0,:])
        break


    