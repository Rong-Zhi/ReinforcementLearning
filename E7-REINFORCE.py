# Continuous mountaincar environment for Monte-Carlo Policy Gradient (REINFORCE)
# Use Gaussian policy with RBF kernel

from gym.envs.classic_control import Continuous_MountainCarEnv
import numpy as np
import itertools

from lib import plotting
import matplotlib.pyplot as plt

env = Continuous_MountainCarEnv()
env.reset()

act = env.action_space.sample()
obv = env.observation_space.sample()
# print('Action space: {0}'.format(act))
# print('Observation Space: ', obv)

# observation = np.array([env.observation_space.sample() for i in range(1000)])

class GP_RBF_kernel(object):
    """
    Guassian Process with RBF features

    """

    def __init__(self, env, features_dim=10, sigma=20, samples=1000):
        self.sigma = sigma
        self.obs_dim = env.observation_space.shape[0]
        self.features_dim = features_dim
        self.features_num = features_dim * self.obs_dim
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.norm_high = np.ones(self.obs_dim) * 1
        self.norm_low = np.ones(self.obs_dim) * -1
        self.centers = np.array([np.linspace(self.norm_low[i], self.norm_high[i], num=features_dim)
                                 for i in range(self.obs_dim)]).T
        self.samples = samples

    # state normalization
    def normalizer(self,state):
        state = (state - self.obs_low) / (self.obs_high - self.obs_low)
        return state * 2 - 1

    # transform state to features
    def transform(self, state):
        state = np.reshape(state,(-1, self.obs_dim))
        norm_state = self.normalizer(state)
        return np.exp(- self.sigma * (norm_state - self.centers)** 2).reshape(self.features_num)

    def plot_1dm(self, ax, x, y):
        for i_feature in range(self.features_dim):
            ax.plot(x, y[:, i_feature])
            ax.set_xlabel('state')
            ax.set_ylabel('RBF features')

    def plot_ndm(self, axes, x, y):
        for i_dim, ax in enumerate(axes.flatten()):
            self.plot_1dm(ax=ax, x=x[:,i_dim], y=y[:,:,i_dim])

    def plot_samples(self, show=True):
        y_features = []
        states = np.array([np.linspace(self.obs_low[i],self.obs_high[i],self.samples)
                           for i in range(self.obs_dim)]).T
        for i_sample, state in enumerate(states):
            features = self.transform(state)
            features = np.reshape(features, newshape=(self.features_dim, self.obs_dim))
            y_features.append(features)
        y_features = np.array(y_features)
        fig, axes = plt.subplots(nrows=self.obs_dim, ncols=1)
        self.plot_ndm(axes=axes, x=states, y=y_features)
        if show is True:
            plt.show()


 # Test RBFkernel

gp = GP_RBF_kernel(env)
# state = env.reset()
# state = np.reshape(state,(-1, gp.obs_dim))
# norm_state = gp.normalizer(state)
# print(gp.centers)
# print(norm_state)
# print(gp.centers - norm_state)
#
# gp.plot_samples(show=True)
# a = 1

class Estimator(object):
    """
    Define parameter estimator for REINFORCE
    """
    def __init__(self, env, alpha):
        self.env = env
        self.gp = GP_RBF_kernel(env)
        self.action_space = env.action_space.shape[0]
        self.state_space = env.observation_space.shape[0]
        self.theta_mu = np.zeros(shape=(self.gp.features_num, self.action_space))
        self.theta_sigma = np.ones(shape=(self.gp.features_num, self.action_space))
        self.alpha = alpha


    def predict(self, state):

        x_s = self.gp.transform(state)
        mu_s = np.dot(self.theta_mu.T, x_s)
        sigma_s = np.exp(np.dot(self.theta_sigma.T, x_s))/10
        action = np.random.normal(mu_s[0],sigma_s[0])
        action = np.atleast_1d(action)
        # action = np.random.multivariate_normal(mean=mu_s,cov=)
        return action, x_s, mu_s, sigma_s

    def update(self,A, mu_s, sigma_s, x_s, G):

        # print(A.shape, mu_s.shape, sigma_s.shape, x_s.shape, G.shape)
        grad_mu = (A - mu_s) * x_s / np.square(sigma_s)
        grad_mu = grad_mu * G
        grad_sigma = (np.square(A - mu_s) / np.square(sigma_s) - 1) * x_s
        grad_sigma = grad_sigma * G
        grad_mu = np.reshape(grad_mu, self.theta_mu.shape)
        self.theta_mu = self.theta_mu +  self.alpha * grad_mu
        # self.theta_sigma += self.alpha * grad_sigma




def discount_norm_rewards(rewards, gamma):
    discounted_rewards = np.zeros(shape=len(rewards))
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    return  discounted_rewards


# estimator = Estimator(env, alpha=0.1)
# state = env.reset()
# for i in range(20):
#     action, xs, ms, ss = estimator.predict(state)
#     next_state, reward, done, _ = env.step(action)
#     print('Action:{0}, X_s:{1}, mu_s{2}, sigma_s{3}, '
#           'Reward:{4}\n'.format(action,xs,ms,ss, reward))





def PolicyGradient(env, estimator, num_episodes, discount_factor=0.9):
    """
    Monte Carlo Policy Gradient

    :param env: OPEN AI enviroment
    :param Estimator: Gaussian policy estimator
    :param num_episodes: Numbers of episodes to run for
    :param discount_factor: Gamma discount factor
    :param learning_rate: Learning rate for Policy Gradient
    :return: An Episode stats object with two numpy arrays for episode length and episode rewards
            A trained Estimator
    """

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        state = env.reset()
        A, mu_s, sigma_s, x_s, R, states = [], [], [], [], [], []
        for t in itertools.count():
            action, xs, ms, ss = estimator.predict(state)
            states.append(state)
            A.append(action)
            mu_s.append(ms)
            sigma_s.append(ss)
            x_s.append(xs)
            next_state, reward, done, _ = env.step(action)
            R.append(reward)
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
#             print('Action:{0}, X_s:{1}, mu_s{2}, sigma_s{3}, '
#                   'Reward:{4}\n'.format(action,xs,ms,ss, reward))
            if done:
                break
            state = next_state

        # G = discount_norm_rewards(R, discount_factor)

        print('Episode: {0}/{1}, Rewards:{2}\n'.format(i_episode, num_episodes, stats.episode_rewards[i_episode]))
        for t in range(len(states)):
            G = sum(r * (discount_factor)**i for i,r in enumerate(R[t:]))

            estimator.update(A=A[t], mu_s=mu_s[t], sigma_s= sigma_s[t], x_s=x_s[t], G=G)
    return stats, estimator

estimator = Estimator(env,alpha=0.1)
stats, estimator = PolicyGradient(env=env,estimator=estimator,num_episodes=500,discount_factor=0.9)

plotting.plot_episode_stats(stats)



state = env.reset()

for t in itertools.count():
    env.render()
    action, _, _, _ = estimator.predict(state = state)
    next_state, reward, done, _ = env.step(action)
    if done:
        break
    state = next_state
