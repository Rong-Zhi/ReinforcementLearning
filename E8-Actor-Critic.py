# Actor Critic for continous mountain car environment

import gym
import sys
import matplotlib
import collections
import sklearn.preprocessing
import sklearn.pipeline
import itertools
import tensorflow as tf
import numpy as np

from lib import plotting
from sklearn.kernel_approximation import RBFSampler
matplotlib.style.use('ggplot')

env = gym.envs.make('MountainCarContinuous-v0')
env. reset()

# print(env.action_space.sample())

# Feature Preprocessing: Normalize to zero mean and unit variance
observe_samples = np.array([env.observation_space.sample() for i in range(10000)])
scalar = sklearn.preprocessing.StandardScaler()
scalar.fit(observe_samples)

# print out the result of normalization
# print("After normalization: ", scalar.transform(observe_samples))

featurizer = sklearn.pipeline.FeatureUnion([
    ('rbf1', RBFSampler(gamma=5, n_components=100)),
    ('rbf2', RBFSampler(gamma=2, n_components=100)),
    ('rbf3', RBFSampler(gamma=1, n_components=100)),
    ('rbf4', RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scalar.transform(observe_samples))

# now scale and featurize the original data
def featurize_state(data):
    scaled = scalar.transform(data.reshape(1,2))
    featurized = featurizer.transform(scaled)
    return featurized[0]

# print(featurize_state(env.observation_space.sample()))


#  Policy improvement -- actor
class Policy_Estimator():
    """
    Policy function approximator, update theta by policy gradient
    """
    def __init__(self, learning_rate=0.01, scope='policy_estimator'):
        self.states = tf.placeholder(tf.float32, [400], name='states')
        self.target = tf.placeholder(dtype=tf.float32, name='targets')
        # Define a linear classifier
        self.mu = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.states,0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.zeros_initializer)

        self.sigma = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.states, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.zeros_initializer)
        self.sigma = tf.squeeze(self.sigma)
        self.sigma = tf.nn.softplus(self.sigma) + 1e-5

        self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = self.normal_dist._sample_n(1)
        self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])

        self.loss = -self.normal_dist.log_prob(self.action) * self.target

        # Add cross entropy loss here for better exploration
        self.loss -= 1e-1 * self.normal_dist.entropy()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.trian_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.action, {self.states:state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = {self.action:action, self.states:state, self.target:target}
        _, loss = sess.run([self.trian_op, self.loss],feed_dict)
        return loss

# Policy evaluation -- critic
# Q function = (state,action) value function
class Value_Estimator():
    """
    Value function approximator, update w by TD(0)
    """
    def __init__(self, learning_rate=0.01, scope='value_estimator'):
        self.states = tf.placeholder(tf.float32, [400], name='states')
        self.target = tf.placeholder(dtype=tf.float32, name='target')
        self.output_layer = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.states,0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.zeros_initializer)
        self.value_estimate = tf.squeeze(self.output_layer)
        self.loss = tf.squared_difference(self.value_estimate, self.target)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.value_estimate,{self.states:state})

    def update(self, state, target,sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = {self.states:state, self.target:target}
        _, loss = sess.run([self.train_op,self.loss], feed_dict)
        return loss

def actor_critic(env, policy_estimator, value_estimator, num_episodes, discount_factor):

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    for i_episode in range(num_episodes):
        state = env.reset()
        # episode = []
        for t in itertools.count():
            action = policy_estimator.predict(state)
            next_state, reward, done, _ = env.step(action)
            # episode.append(Transition(state, action, reward, next_state, done))

            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward

            next_value = value_estimator.predict(next_state)
            td_target = reward + discount_factor * next_value
            td_error = td_target - value_estimator.predict(state)
            
            value_estimator.update(state, td_target)

            # using td error as advantage function
            policy_estimator.update(state, td_error, action)

            if done:
                break
            state = next_state
        print('Episode: {0}/{1}, Rewards:{2}\n'.format(i_episode, num_episodes, stats.episode_rewards[i_episode]))
    return policy_estimator,value_estimator, stats


tf.reset_default_graph()
global_step = tf.Variable(0, name='global_step', trainable=False)
policy_estimator = Policy_Estimator(learning_rate=0.01)
value_estimator = Value_Estimator(learning_rate=0.1)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    actor, critic, stats = actor_critic(env=env, policy_estimator=policy_estimator, value_estimator=value_estimator, num_episodes=50, discount_factor=0.95)
    plotting.plot_episode_stats(stats)


