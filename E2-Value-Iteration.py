import numpy as np
import pprint
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import  savefig
import warnings
# if "../" not in sys.path:
#   sys.path.append("../")
from lib.envs.gridworld import GridworldEnv
from lib.envs.proGenerator import *


pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()





def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    while True:
        delta = 0
        for s in range(env.nS):
            v = np.zeros(env.nA)
            # Look at all possible next actions
            for a, action_prob in enumerate(policy[s]):
                # for each action, look at the possible next states:
                for prob, next_state, reward, done in env.P[s][a]:
                    # calculate the (expected)value function
                    v[a] += prob * (reward + discount_factor * V[next_state])
            best_action_v = np.max(v)
            delta = max(delta, np.abs(best_action_v - V[s]))
            V[s] = best_action_v
        if delta < theta:
            break
    for s in range(env.nS):
        v = np.zeros(env.nA)
        # Look at all possible next actions
        for a, action_prob in enumerate(policy[s]):
            # for each action, look at the possible next states:
            for prob, next_state, reward, done in env.P[s][a]:
                # calculate the (expected)value function
                v[a] += prob * (reward + discount_factor * V[next_state])
        best_action = np.argmax(v)
        policy[s, best_action] = 1
    return policy, V

policy, V = value_iteration(env)
print("Policy Probability Distribution:")
print(policy)
print("")


# reshape the grid policy by argmax(policy) along each state
print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Reshaped Grid Value Function:")
print(V.reshape(env.shape))
print("")