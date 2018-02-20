import gym
import numpy as np
import collections
from collections import defaultdict
import sklearn.pipeline
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler

env = gym.envs.make('Cart-Pole-v0')

def single_path(env, policy, path_length, discount_factor):
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    V = defaultdict(lambda: np.zeros(e))
    state = env.reset()
    samples = []
    for t in range(path_length):
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(action)
        samples.append(Transition(state, action, reward, next_state, done))
        state = next_state
    t = 0
    for i_sample in samples:
        G = sum(r * (discount_factor) ** i for i, r in enumerate(i_sample.reward[t:]))
        if t == 0:
            eta_policy = G/path_length
        Q[i_sample.state][i_sample.action] = G / (path_length - 1)
        t += 1

    return samples, Q, eta_policy

def vine(env):
    pass

def advantage(env):
    pass

def pg_estimator(env):
    pass

def conjugate_gradient(env):
    pass

def policy_update(env):
    pass

class Agent(object):
    def __init__(self,
         env,
         policy_iteration=200,
         stepsize=0.01,
         discount=0.99,
         path_num=50,
         path_length=500,
         hidden_sizes = 30):

        self.act_space = env.action_space.n
        self.states = tf.placeholder(tf.float32, [env.observation_space[0]] ,name='states')
        self.adv = tf.placeholder(dtype=tf.float32, name='advantages')
        self.layer1 = tf.layers.dense(self.states, hidden_sizes)
        self.output = tf.layers.dense(self.layer1, self.act_space)
        self.action_prob = tf.nn.softmax(self.output)



def trpo():
    pass