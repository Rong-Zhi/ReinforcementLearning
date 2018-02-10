import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
from collections import defaultdict

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

from gym.envs.classic_control import MountainCarEnv

# matplotlib.style.use('ggplot')

env = gym.envs.make('MountainCar-v0')
# env = MountainCarEnv()
env.reset()

# count = 0
# while True:
#     count += 1
#     action = env.action_space.sample()
#     next_state, reward, done, _ = env.step(action)
#     if done:
#         break
#     env.render()
#     print(count)

print('Action space dimention:',env.action_space.n)

observation_samples = np.array([env.observation_space.sample() for x in range(1000)])


# Feature Preprocessing: Normalize to zero mean and unit variance
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_samples)

# Used to convert a state to a featurized representation.
# We use RBF kernels with different variances to cover different parts of the spacef
featurizer = sklearn.pipeline.FeatureUnion([
    ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
    ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
    ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
    ('rbf4', RBFSampler(gamma=0.5, n_components=100))
])

featurizer.fit(scaler.transform(observation_samples))


class Estimator():
    """
    Value Function approximator.
    """

    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
        """

        # TODO: Implement this!
        features = self.featurize_state(s)

        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        # TODO: Implement this!
        features = self.featurize_state(s)

        self.models[a].partial_fit([features],[y])

        return None


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))


    for i_episode in range(num_episodes):
        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
        sys.stdout.flush()

        # TODO: Implement this!
        state = env.reset()


        for t in itertools.count():
            # same as before
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)),p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # update stats
            stats.episode_lengths[i_episode] = t
            stats.episode_rewards[i_episode] += reward

            q_values_next = estimator.predict(next_state)


            td_target = reward + discount_factor * np.max(q_values_next)

            estimator.update(state, action, td_target)
            if done:
                break

            state = next_state



    return stats, estimator

estimator = Estimator()

# # Note: For the Mountain Car we don't actually need an epsilon > 0.0
# # because our initial estimate for all states is too "optimistic" which leads
# # to the exploration of all states.


stats,estimator = q_learning(env, estimator, 100, epsilon=0.0)

# plotting.plot_cost_to_go_mountain_car(env, estimator)
# plotting.plot_episode_stats(stats, smoothing_window=25)


state = env.reset()
policy = make_epsilon_greedy_policy(estimator, 0.0, env.action_space.n)

for t in itertools.count():
    env.render()
    action_probs = policy(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    next_state, reward, done, _ = env.step(action)
    if done:
        break
    state = next_state

