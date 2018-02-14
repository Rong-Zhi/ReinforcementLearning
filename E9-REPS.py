# REPS for discrete Nchains env
# Gaussian Policy with linear mean
# Using RBF features for state

import gym
import itertools
import collections
import matplotlib

from lib import plotting
import numpy as np
import scipy.optimize
from scipy.misc import logsumexp


env = gym.envs.make('NChain-v0')


def featurizer(data, inverse=False):
    N = env.observation_space.shape[0]
    one_hot = np.eye(N)
    if inverse is False:
        return np.reshape(one_hot[data],(-1,1))
    else:
        return data.argmax()


class REPS():
    """
    Relative Entropy Policy Search

    """
    def __init__(self, epsilon=0.5, optimzer=scipy.optimize.fmin_l_bfgs_b, min_eta=1e-8):
        self.optmizer = optimzer
        self.epsilon = epsilon
        self.act_space = env.action_space.shape[0]
        self.state_space = env.action_space.shape[0]
        self.bellman_error = np.zeros((self.state_space, self.act_space))
        self.feature_difference =np.zeros((self.state_space, self.act_space))
        self.min_eta = min_eta
        self.samples = []

    def samplefromenv(self, env, policy):
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        state = env.reset()
        for i in itertools.count():
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            self.samples.append(Transition(state, action, reward, next_state, done))
            if done:
                break
            state = next_state

    def preknowledge(self, eta, theta):
        n_tderror = np.zeros((self.state_space, self.act_space))
        n_difference = np.zeros((self.state_space, self.act_space))
        n_appearence = np.zeros((self.state_space, self.act_space))

        for i in self.samples:
            i_state_feature = featurizer(i.state)
            i_nextstate_feature = featurizer(i.next_state)
            tmp = np.dot(i_state_feature.T, theta) - np.dot(i_nextstate_feature.T, theta)
            n_tderror[i.state][i.action] += i.reward + tmp[0]
            n_difference[i.state][i.action] += i_nextstate_feature - i_state_feature
            n_appearence[i.state][i.action] += 1

        self.bellman_error = np.divide(n_tderror, n_appearence,
                                  out=np.zeros_like(n_tderror), where=n_appearence != 0)
        self.feature_difference = np.divide(n_difference, n_appearence,
                                       out=np.zeros_like(n_tderror), where=n_appearence != 0)

    def dual_function(self, params):
        eta = params[0]
        theta = np.array(params[1:])
        theta = theta.reshape(self.state_space, 1)
        self.preknowledge(eta, theta)
        g = eta * logsumexp((self.epsilon + 1/eta * self.bellman_error)/len(self.samples))

        return g

    def dual_grad(self, params):
        eta = params[0]
        theta = np.array(params[1:])
        theta = theta.reshape(self.state_space, 1)

        dg_eta = eta * np.sum(np.exp(self.epsilon + 1/eta * self.bellman_error) * self.feature_difference)\
                 /np.sum(np.exp(self.epsilon + 1/eta * self.bellman_error))
        dg_theta = logsumexp(self.epsilon + 1/eta * self.bellman_error)\
                   - np.sum(np.exp(self.epsilon + 1/eta * self.bellman_error) * self.bellman_error / eta**2)\
                   /np.sum(np.exp(self.epsilon + 1/eta * self.bellman_error))
        return np.hstack([dg_eta, dg_theta])

    def dual_optimize(self, init_eta, init_theta):
        init_params = np.hstack([init_eta, init_theta])
        result,_,_ = self.optmizer(self.dual_function, init_params, fprime=self.dual_grad)
        return result[0],result[1:]


    def update(self, old_policy, eta, theta):

        bellman_error, _ = self.preknowledge(eta, theta)
        new_policy = old_policy * np.exp(bellman_error/eta) \
                     / (np.sum(old_policy * np.exp(bellman_error)/eta))
        return new_policy



    def policy_search(self,env, policy, num_updates, init_eta, init_theta):

        for k in range(num_updates):
            self.samplefromenv(env,policy)
            eta,theta = self.dual_optimize(init_eta, init_theta)
            new_policy = self.update(policy, eta, theta)
            policy = new_policy





