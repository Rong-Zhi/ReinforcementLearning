# REPS for discrete Nchains env
# Gaussian Policy with linear mean
# Using RBF features for state

import gym
import itertools
import collections
import matplotlib

from collections import defaultdict
from lib import plotting
import numpy as np
from gym.envs.toy_text.nchain import NChainEnv
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
env = NChainEnv(n=5)


def featurizer(data, inverse=False):
    N = env.observation_space.shape[0]
    one_hot = np.eye(N)
    if inverse is False:
        return np.reshape(one_hot[data],(1,N))
    else:
        return data.argmax()


class REPS():
    """
    Relative Entropy Policy Search

    """
    def __init__(self, epsilon=0.5, min_eta=1e-8):
        self.epsilon = epsilon
        self.act_space = env.action_space.shape[0]
        self.state_space = env.observation_space.shape[0]
        self.policy = np.ones((self.state_space,self.act_space))/2
        self.sa_pair = [(a,b) for a in range(self.state_space) for b in range(self.act_space)]
        self.samples = []
        self.min_eta = min_eta

    def samplefromenv(self, env):
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        state = env.reset()
        self.samples = []
        R = 0.
        for i in range(500):
            action_prob = self.policy[state]
            action = np.random.choice(len(action_prob),p=action_prob)
            next_state, reward, done, _ = env.step(action)
            self.samples.append(Transition(state, action, reward, next_state, done))
            R += reward
            state = next_state
        return R/i

    def preknowledge(self, eta, theta):
        n_tderror, n_difference, n_appearence = defaultdict(float), defaultdict(float), defaultdict(float)
        bellman_error = []
        feature_difference = []
        visit_time = []
        for i in self.samples:
            i_state_feature = featurizer(i.state)
            i_nextstate_feature = featurizer(i.next_state)
            state_action = (i.state, i.action)
            tmp =  np.dot(theta, i_nextstate_feature.T) - np.dot(theta, i_state_feature.T)
            n_tderror[state_action] += i.reward + tmp[0]
            n_difference[state_action] += i_nextstate_feature - i_state_feature
            n_appearence[state_action] += 1
        self.sa_keys = []
        for key in sorted(n_tderror.keys()):
            self.sa_keys.append(key)
            bellman_error.append(n_tderror[key] / n_appearence[key])
            feature_difference.append(n_difference[key]/n_appearence[key])
            visit_time.append(n_appearence[key])
        visit_time = np.expand_dims(np.array(visit_time),axis=1)
        return np.array(bellman_error), np.squeeze(np.array(feature_difference)), visit_time

    def dual_function(self, params):
        eta = params[0]
        theta = np.array(params[1:])
        theta = theta.reshape(1, self.state_space)
        bellman_error, feature_difference, visit_time = self.preknowledge(eta, theta)
        # Z = np.exp(self.epsilon + bellman_error * 1.0/eta)
        # g = eta * np.log(np.average(Z, weights=visit_time))
        max_bl = np.max(bellman_error)
        Z = np.exp((bellman_error-max_bl)/eta)
        g = eta*self.epsilon + max_bl + eta * np.log(np.average(Z,weights=visit_time))
        # dg_theta = np.average(feature_difference, axis=0, weights=(Z * visit_time).reshape(Z.shape[0]))
        # dg_eta = self.epsilon + np.log(np.average(Z, weights=visit_time)) - np.average((bellman_error - max_bl), axis=0,
        #                                                                                    weights=Z * visit_time) / eta
        return g

    def dual_grad(self, params):
        eta = params[0]
        theta = np.array(params[1:])
        theta = theta.reshape(1, self.state_space)
        bellman_error, feature_difference, visit_time = self.preknowledge(eta, theta)
        # Z = np.exp(self.epsilon + 1.0/eta * bellman_error)
        # dg_theta = eta * np.sum(Z * feature_difference * visit_time, axis=0)/np.sum(Z * visit_time)
        # dg_eta = np.log(np.sum(Z * visit_time))- np.sum(Z * visit_time/ eta**2)/np.sum(Z * visit_time)
        max_bl = np.max(bellman_error)
        Z = np.exp((bellman_error - max_bl) / eta)
        dg_theta = np.average(feature_difference,axis=0,weights=(Z*visit_time).reshape(Z.shape[0]))
        dg_eta = self.epsilon + np.log(np.average(Z,weights=visit_time)) - np.average((bellman_error-max_bl), axis=0, weights=Z*visit_time) / eta
        return np.hstack([dg_eta, dg_theta])


    def dual_optimize(self, init_eta, init_theta):
        init_params = np.hstack((init_eta, init_theta))
        bounds = [(-np.inf, np.inf) for _ in init_params]
        bounds[0] = (1e-5, np.inf)
        result, fun, _ = fmin_l_bfgs_b(self.dual_function, init_params, fprime=self.dual_grad, bounds=bounds)
        # result = minimize(self.dual_function, x0=init_params, method='slsqp',jac=True, bounds=bounds)
        # print(result.fun)
        # if result.success:
        #     print("Optimization success! \n")
        return result[0],result[1:], fun


    def update(self, eta, theta, fun):
        bellman_error, _, _ = self.preknowledge(eta,theta)
        adv_sa = fun * np.ones((self.state_space, self.act_space))
        adv_sa[tuple(zip(*self.sa_keys))] = bellman_error
        Z = np.exp((adv_sa - np.max(adv_sa)) / eta)
        pi_new = np.copy(self.policy)
        pi_new *= Z
        pi_new /= np.sum(pi_new, axis=1,keepdims=True)
        self.policy = pi_new
        return self.policy


    def policy_search(self, env, num_updates, init_eta, init_theta):
        for k in range(num_updates):
            R = self.samplefromenv(env)
            eta, theta, fun = self.dual_optimize(init_eta, init_theta[0])
            self.policy = self.update(eta, theta, fun)
            print('Episode: {0}/{1}, Reward:{2}, Policy:{3}\n'.format(k, num_updates, R, self.policy))
        return self.policy

reps = REPS()
reps.policy_search(env=env, num_updates=30, init_eta=15.0 , init_theta=np.ones((1, env.observation_space.shape[0])))


