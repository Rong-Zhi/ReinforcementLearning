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


# Define Policy evalutation
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # TODO: Implement!
        for s in range(env.nS):
            v = 0
            # Look at all possible next actions
            for a, action_prob in enumerate(policy[s]):
                # for each action, look at the possible next states:
                for prob, next_state, reward, done in env.P[s][a]:
                    # calculate the (expected)value function
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return np.array(V)


# Define Policy improvement
def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        # Implement this!
        # evaluate current policy
        V = policy_eval_fn(policy, env, discount_factor)
        # will be set to false if the policy changes
        policy_stable = True

        for s in range(env.nS):
            # take the action with under 'current policy'
            chosen_a = np.argmax(policy[s])

            # find the best action by one-step look ahead
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)
            # greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]

        if policy_stable:
            return policy, V

policy, V = policy_improvement(env)
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