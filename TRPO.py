import gym
import numpy as np
import sklearn.pipeline
import tensorflow as tf
from sklearn.kernel_approximation import RBFSampler

env = gym.envs.make('Cart-Pole-v0')

observe_samples = np.array([env.observation_space.sample() for i in range(10000)])

featurizer = sklearn.pipeline.FeatureUnion([
    ('rbf1', RBFSampler(gamma=5, n_components=100)),
    ('rbf2', RBFSampler(gamma=2, n_components=100)),
    ('rbf3', RBFSampler(gamma=1, n_components=100)),
    ('rbf4', RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(observe_samples)

def featurize_state(data):
    featurized = featurizer.transform(data.reshape(1,env.observation_space.shape[0]))
    return featurized[0]

def single_path(env, path_num, path_length):
    pass

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

def TRPO(env,
         policy_iteration=200,
         stepsize=0.01,
         discount=0.99,
         path_num=50,
         path_length=1000,
         computation_time=5,
         hidden_sizes = (32, 32)):
    pass

