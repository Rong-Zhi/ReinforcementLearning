# REPS for discrete Nchains env
# Gaussian Policy with linear mean
# Using RBF features for state

import gym
import itertools
import collections

import tensorflow as tf
import matplotlib

from lib import plotting
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sklearn.pipeline
import scipy.optimize
from sklearn.kernel_approximation import RBFSampler


env = gym.envs.make('NChain-v0')
enc = OneHotEncoder()

observe_samples = np.array([env.observation_space.sample() for i in range(10000)])
enc.fit(observe_samples.reshape(10000,1))

# def featurizer_state(data):
#
#     return featurized[0].reshape(-1,1)


class Critic():
    """
    Critic part for REPS -- policy evaluation

    """
    def __init__(self, epsilon=0.5, optimzer=scipy.optimize.fmin_l_bfgs_b):
        self.optmizer = optimzer
        self.epsilon = 0.5
        self.act_space = env.action_space.n
        self.num_tderror = collections.defaultdict(lambda: np.zeros(self.act_space))
        self.num_difference = collections.defaultdict(lambda: np.zeros(self.act_space))
        self.num_appearence = collections.defaultdict(lambda: np.zeros(self.act_space))
        self.delta = collections.defaultdict(lambda: np.zeros(self.act_space))

        self.theta = np.ones(shape=(400,1))


    def preknowledge(self, samples, theta):
        # for i in samples:
        #     i_state_feature = featurizer_state(i.state)
        #     i_nextstate_feature =featurizer_state(i.next_state)
        #     tmp = np.dot(i_state_feature.T, theta)- np.dot(i_nextstate_feature.T, theta)
        #     self.num_tderror[i.state][i.action] += i.reward + tmp[0]
        #     self.num_difference[i.state][i.action] += i_nextstate_feature -i_state_feature
        #     self.num_appearence[i.state][i.action] += 1
        pass


    def predict(self):

        for state, action in self.num_tderror.items():
            self.delta[state][action] = self.num_tderror[state][action]

        pass

    def update(self):
        pass

class Actor():
    """
    Actor par for REPS -- policy improvement

    """
    def __init__(self):

        pass

    def predict(self):
        pass

    def update(self):
        pass


def samplefromenv(env, policy):
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    samples = []
    state = env.reset()
    for i in itertools.count():
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        samples.append(Transition(state, action, reward, next_state, done))
        if done:
            break
        state = next_state
    return samples

def reps(env, critic, actor, num_updates):

    pass





env.reset()