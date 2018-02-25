import numpy as np
import tensorflow as tf
from gym.envs.classic_control import Continuous_MountainCarEnv
from gym.envs.classic_control import PendulumEnv
from lib import plotting
from collections import namedtuple
import matplotlib

matplotlib.style.use('ggplot')

class Policy_net(object):
    """
    Actor net: build neural network with 2 hidden layers,
                each layer contains 64 neurons,
                output of network is mu and sigma of gaussian distribution,

                learning_rate = 0.0001
                epsilon = 0.2
    """
    def __init__(self):
        self.epsilon = 0.2
        self.learning_rate = 0.0001
        self.act_space = env.action_space.shape[0]
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high
        self.state_space = env.observation_space.shape[0]

        # placeholder
        self.states = tf.placeholder(tf.float32, shape=[None, self.state_space], name='states')
        self.test_action = tf.placeholder(tf.float32, shape=[None, self.act_space], name='test_action')
        self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')
        self.old_log_dist = tf.placeholder(tf.float32, shape=[None, 1], name='old_log_distribution')

        # layers
        self.l1 = tf.layers.dense(inputs=self.states, units=64, activation=tf.nn.relu)
        # self.l2 = tf.layers.dense(self.l1, 64, activation=tf.nn.tanh)

        # outputs
        self.mu = tf.layers.dense(inputs=self.l1, units=self.act_space, activation=tf.identity)
        # self.sigma = tf.squeeze(tf.layers.dense(self.l1, self.act_space, activation=tf.nn.tanh))
        # self.sigma = tf.nn.softplus(self.sigma) + 1e-5

        self.sigma = tf.Variable(tf.ones([1, self.act_space]))
        self.sigma = tf.exp(self.sigma)

        # action probability and action
        self.new_policy_dist = tf.contrib.distributions.MultivariateNormalDiag(self.mu, self.sigma)
        self.action = self.new_policy_dist.sample(self.act_space)
        self.action = tf.clip_by_value(self.action, self.act_low, self.act_high)

        # intermediate result
        self.new_log_dist = self.new_policy_dist.log_prob(self.test_action)
        ratio = tf.exp(self.new_log_dist - self.old_log_dist)
        clip_p = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon)

        # loss and optimizer
        self.surrogate_loss = -tf.reduce_mean(tf.minimum(tf.multiply(ratio, self.advantage),
                                                         tf.multiply(clip_p, self.advantage)),
                                              name='surrogate_loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.surrogate_loss, global_step=tf.contrib.framework.get_global_step())

    def predict_action(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return np.squeeze(sess.run(self.action, {self.states:state}), axis=0)

    def predict_log_dist(self, state, action, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.new_log_dist, {self.states:state, self.test_action:action})

    def update(self, state, advantage, test_action, old_policy_dist, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states:state, self.advantage: advantage,
                     self.test_action:test_action,
                     self.old_log_dist:old_policy_dist}
        _ ,loss = sess.run([self.train_op, self.surrogate_loss],
                                        feed_dict=feed_dict)
        return loss




class Value_net(object):
    """
    Critic net: build neural network with 2 hidden layers,
                each layer contains 64 neurons,
                learning rate = 0.0002
                optimizer -- Adamoptimizer
    """
    def __init__(self):
        self.learning_rate = 0.0002
        self.act_space = env.action_space.shape[0]
        self.state_space = env.observation_space.shape[0]

        # placeholder
        self.states = tf.placeholder(tf.float32, shape=[None, self.state_space], name='states')
        self.target = tf.placeholder(tf.float32, shape=[None, 1], name='target')

        # layers
        self.l1 = tf.layers.dense(inputs=self.states, units=64, activation=tf.nn.tanh)
        # self.l2 = tf.layers.dense(self.l1, 64, activation=tf.nn.tanh)

        # output
        self.output = tf.layers.dense(inputs=self.l1, units=self.act_space, activation=tf.identity)
        self.value_estimate = tf.squeeze(self.output)

        # loss and optimizer
        self.loss = tf.squared_difference(self.value_estimate, self.target, name='loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.states : state})


    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss],
                                        feed_dict=feed_dict)
        return loss


def rollout(env, policy, state_space):
    state = env.reset()
    state = state.reshape(1, state_space)
    done = False
    while not done:
        action = policy.predict_action(state)
        print(action)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.reshape(1, state_space)
        yield state, action, next_state, reward
        state = next_state


def rollouts(env, policy, timestep, state_space):
    Transition = namedtuple("Transition", ["state", "action", "next_state", "reward"])
    paths = []
    num_rollout = 0
    while len(paths) < timestep:
        for trans in rollout(env, policy, state_space):
            state, action, next_state, reward = trans
            paths.append(Transition(state, action, next_state, reward))
        num_rollout += 1

    return paths, num_rollout


def compute_advantage(value_func, states, reward, discount_factor):
    values = value_func.predict(states)
    target = [reward[i] + discount_factor * values[i+1] - values[i] for i in range(len(states) - 1)]
    advantage = [sum([t * discount_factor**i for i,t in enumerate(target[n:])]) for n in range(len(states)-1)]
    return np.expand_dims(advantage, axis=-1), np.expand_dims(target, axis=-1)


def ppo(env, policy_estimator, value_estimator):

    discount_factor = 0.99
    num_iteration = 100
    timestep = 500
    batchsize = 32
    epoch_per_iter = 15

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_iteration),
        episode_rewards=np.zeros(num_iteration))

    for i_iteration in range(num_iteration):

        paths, num_rollout = rollouts(env, policy_estimator, timestep, env.observation_space.shape[0])
        states, actions, next_states, reward = map(np.array, zip(*paths))
        states = np.squeeze(states)
        actions = np.squeeze(actions, axis=(1,))

        print('Training {0}/{1}, reward:{2} \n'.format(i_iteration, num_iteration, np.sum(reward)/num_rollout))

        stats.episode_lengths[i_iteration] = i_iteration
        stats.episode_rewards[i_iteration] = np.sum(reward)/num_rollout

        T = len(paths)
        # update value estimator
        for epoch in range(epoch_per_iter):
            advantage, target = compute_advantage(value_estimator, states, reward, discount_factor)
            for i in range(0, T-2, batchsize):
                end = min(i+batchsize, T-2)
                lossv = value_estimator.update(states[i:end], target[i:end])


        # update policy estimator
        advantage, target = compute_advantage(value_estimator, states, reward, discount_factor)
        old_log_dist = policy_estimator.predict_log_dist(states, actions)

        for epoch in range(epoch_per_iter):
            for i in range(0, T-2, batchsize):
                end = min(i+batchsize, T-2)
                lossp = policy_estimator.update(states[i:end], actions[i:end],
                                        advantage[i:end], old_log_dist[i:end])

        # print('Loss: Value estimator-{0}, Policy estimator-{1}'.format(lossv, lossp))

    return stats, policy_estimator, value_estimator


# env = Continuous_MountainCarEnv()
env = PendulumEnv()
tf.reset_default_graph()
global_step = tf.Variable(0, name='global_step', trainable=False)

policy_estimator = Policy_net()
value_estimator = Value_net()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    stats, policy_estimator, value_estimator = ppo(env, policy_estimator, value_estimator)
    plotting.plot_episode_stats(stats)
    #
    # state = env.reset()
    # while True:
    #     env.render()
    #     action = policy_estimator(sess, state, epsilon=1.0)
    #     next_state, reward, done, _ = env.step(action)
    #     if done:
    #         break
    #     state = next_state