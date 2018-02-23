import gym
import numpy as np
import collections
from collections import defaultdict
from scipy.optimize import fmin_cg
from scipy.optimize import line_search
import tensorflow as tf


env = gym.envs.make('Cart-Pole-v0')

def single_path(env, policy, path_length, discount_factor):
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
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


class Policy_net(object):

    def __init__(self, env, policy_iteration=40, stepsize=0.01,
                 discount=0.99, path_num=50, path_length=100, hidden_sizes = (32,3)):

        # Initialize parameters
        self.act_space = env.action_space.n
        self.state_space = env.observation_space.shape[0]
        self.policy_iteration = policy_iteration
        self.stepsize = stepsize
        self.path_num = path_num
        self.path_length = path_length
        self.discount = discount
        self.hidden_size = hidden_sizes


    def build(self):

        # tensorflow placeholder
        self.states = tf.placeholder(tf.float32, shape=[self.state_space] ,name='states') # modify shape later
        self.action = tf.placeholder(tf.int64, shape=[None] ,name='action')
        self.adv = tf.placeholder(tf.float32,shape=[None], name='advantages')
        self.prev_policy_dist = tf.placeholder(tf.float32, shape=[None, self.act_space], name='prev_policy')

        # neural network
        self.layer1 = tf.layers.dense(self.states, self.hidden_size[0], activation=tf.nn.relu)
        self.layer2 = tf.layers.dense(self.layer1, self.hidden_size[1], activation=tf.nn.relu)

        # output
        self.output = tf.layers.dense(self.layer2, self.act_space)
        self.new_policy_dist = tf.squeeze(tf.nn.softmax(self.output))

        # intermediate result
        self.new_policy_pi = tf.gather(self.new_policy_dist, self.action)
        self.old_policy_pi = tf.gather(self.prev_policy_dist, self.action)

        # loss
        self.surrogate_loss = -tf.reduce_mean(tf.div(
                self.new_policy_pi, self.old_policy_pi) * self.adv, name='surrogate_loss')

        self.entropy_loss = tf.reduce_mean(-self.new_policy_dist *
                                           tf.log(self.new_policy_dist), axis=0, name='entropy_loss')

        self.kl_loss = tf.reduce_mean(self.prev_policy_dist * tf.log(
            tf.div(self.prev_policy_dist, self.new_policy_dist)), axis=0, name='KL_loss')

        



        self.train_op = tf.train.AdamOptimizer().minimize(self.surrogate_loss)



    def predict(self, sess, state):
        return sess.run(self.new_policy_dist , { self.states: state })

    def update(self, sess, state, advant, prev_policy):
        feed_dict = {self.states: state, self.adv: advant,
                      self.prev_policy_dist: prev_policy}
        global_step, _, loss = sess.run(
            [tf.contrib.framework.get_global_step(), self.train_op, self.surrogate_loss],
            feed_dict)
        return loss

class Value_net(object):
    def __init__(self):


        pass

