"""report bugs to riad@robot-learning.de"""
import gym
from gym.envs.classic_control import Continuous_MountainCarEnv
import tensorflow as tf
import numpy as np
from lib import plotting

class MLP:
    def __init__(self, sizes, activations=None):
        if activations is None:
            activations = [tf.nn.relu] * (len(sizes) - 2) + [tf.identity]
        self.x = last_out = tf.placeholder(dtype=tf.float32, shape=[None, sizes[0]])
        for l, size in enumerate(sizes[1:]):
            last_out = tf.layers.dense(last_out, size, activation=activations[l], kernel_initializer=tf.glorot_uniform_initializer())
        self.out = last_out


class MLPGaussianPolicy:
    def __init__(self, session, sizes, activations=None, init_sigma=1.):
        self.mlp = MLP(sizes, activations)
        self.sess = session

        # action tensor (diagonal Gaussian)
        act_dim = sizes[-1]
        self.logsigs = tf.Variable(np.log(init_sigma) * tf.ones([1, act_dim]))
        self.sigs = tf.exp(self.logsigs)
        self.gaussPol = tf.contrib.distributions.MultivariateNormalDiag(self.mlp.out, self.sigs)
        self.act_tensor = self.gaussPol.sample()

        # action proba (diagonal Gaussian)
        self.test_action = tf.placeholder(dtype=tf.float32, shape=[None, act_dim])
        self.log_prob = tf.expand_dims(self.gaussPol.log_prob(self.test_action), axis=-1)  # expand vector returned by log_prob to row vector

        # pol entropy (for logging only)
        self.entropy = tf.reduce_sum(self.logsigs) + act_dim * np.log(2 * np.pi * np.e) / 2

    def get_action(self, obs):
        return np.squeeze(self.sess.run(self.act_tensor, {self.mlp.x: np.asmatrix(obs)}), axis=0)

    def get_log_proba(self, obs, act):
        return self.sess.run(self.log_prob, {self.mlp.x: obs, self.test_action: act})


class PPO:
    def __init__(self, session, policy, vfunc, e_clip=.2, a_lrate=5e-4, v_lrate=5e-4):
        self.sess = session
        self.pol = policy
        self.v = vfunc

        # loss for v function
        self.target_v = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.loss_v = tf.losses.mean_squared_error(self.v.out, self.target_v)
        self.optimizer_v = tf.train.AdamOptimizer(v_lrate).minimize(self.loss_v)

        # clip loss for policy update
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.old_log_probas = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        proba_ratio = tf.exp(self.pol.log_prob - self.old_log_probas)
        self.clip_pr = tf.clip_by_value(proba_ratio, 1 - e_clip, 1 + e_clip)
        self.neg_objective_act = -tf.reduce_mean(tf.minimum(tf.multiply(proba_ratio, self.advantage), tf.multiply(self.clip_pr, self.advantage)))
        self.optimizer_act = tf.train.AdamOptimizer(a_lrate).minimize(self.neg_objective_act)

    def train_v(self, obs, target_v):
        self.sess.run(self.optimizer_v, {self.v.x: obs, self.target_v: target_v})

    def evaluate_v(self, obs, target_v):
        return self.sess.run(self.loss_v, {self.v.x: obs, self.target_v: target_v})

    def train_pol(self, obs, old_act, old_log_probas, advantages):
        return self.sess.run(self.optimizer_act, {self.pol.mlp.x: obs, self.pol.test_action: old_act,
                                                  self.advantage: advantages, self.old_log_probas: old_log_probas})

    def evaluate_pol(self, obs, old_act, old_log_probas, advantages):
        return -self.sess.run(self.neg_objective_act, {self.pol.mlp.x: obs, self.pol.test_action: old_act,
                                                       self.advantage: advantages, self.old_log_probas: old_log_probas})


def next_batch_idx(batch_size, data_set_size):
    batch_idx_list = np.random.choice(data_set_size, data_set_size, replace=False)
    for batch_start in range(0, data_set_size, batch_size):
        yield batch_idx_list[batch_start:min(batch_start + batch_size, data_set_size)]


def rollout(env, policy, render=False):
    # Generates transitions until episode's end
    obs = env.reset()
    done = False
    while not done:
        if render:
            env.render()
        act = policy(obs)
        nobs, rwd, done, _ = env.step(np.minimum(np.maximum(act, env.action_space.low), env.action_space.high))
        yield obs, act, rwd, done
        obs = nobs


def rollouts(env, policy, min_trans, render=False):
    # Keep calling rollout and saving the resulting path until at least min_trans transitions are collected
    keys = ["obs", "act", "rwd", "done"]  # must match order of the yield above
    paths = {}
    for k in keys:
        paths[k] = []
    nb_paths = 0
    while len(paths["rwd"]) < min_trans:
        for trans_vect in rollout(env, policy, render):
            for key, val in zip(keys, trans_vect):
                paths[key].append(val)
        nb_paths += 1
    for key in keys:
        paths[key] = np.asarray(paths[key])
        if paths[key].ndim == 1:
            paths[key] = np.expand_dims(paths[key], axis=-1)  # force all entries to be matrices
    paths["nb_paths"] = nb_paths
    return paths


def get_gen_adv(paths, v_func, discount, lam):
    # computes generalized advantages from potentially off-policy data
    v_values = v_func(paths["obs"])
    gen_adv = np.empty_like(v_values)
    for rev_k, v in enumerate(reversed(v_values)):
        k = len(v_values) - rev_k - 1
        if paths["done"][k]:  # this is a new path. always true for rev_k == 0
            gen_adv[k] = paths["rwd"][k] - v_values[k]
        else:
            gen_adv[k] = paths["rwd"][k] + discount * v_values[k + 1] - v_values[k] + discount * lam * gen_adv[k + 1]
    return gen_adv, v_values


def main():
    # seeding, comment out if you do not want deterministic behavior across runs
    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # env = gym.make('MountainCarContinuous-v0')
    env = gym.make('BipedalWalker-v2')
    # env = gym.make('BipedalWalkerHardcore-v2')
    # env = gym.make('LunarLanderContinuous-v2')
    # env = Continuous_MountainCarEnv()
    env.seed(seed)

    # params
    nb_iter = 100  # one iter -> at least min_trans_per_iter generated
    min_trans_per_iter = 3200
    render_every = 100
    epochs_per_iter = 15  # for updating the v function and the policy
    exploration_sigma = 1.5
    discount = .99
    lam_trace = .95
    e_clip = .2  # the 'step size'
    batch_size = 64

    # mlp for v-function and policy and ppo
    session = tf.Session()
    layer_sizes = [env.observation_space.shape[0]] + [64] * 2
    layer_activations = [tf.nn.relu] * (len(layer_sizes) - 2) + [tf.nn.tanh]
    policy = MLPGaussianPolicy(session, layer_sizes + [env.action_space.shape[0]], layer_activations + [tf.nn.tanh], exploration_sigma)
    vmlp = MLP(layer_sizes + [1], layer_activations + [tf.identity])
    get_v = lambda obs: session.run(vmlp.out, {vmlp.x: obs})
    ppo = PPO(session, policy, vmlp, e_clip=e_clip)
    session.run(tf.global_variables_initializer())

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(nb_iter),
        episode_rewards=np.zeros(nb_iter))

    for it in range(nb_iter):
        print('-------- iter ', it, ' --------')
        # if (it + 1) % render_every == 0:  # does not render at first iteration
        #     render = True
        # else:
        #     render = False
        render = False
        # Generates transition data by interacting with the env
        paths = rollouts(env, policy=policy.get_action, min_trans=min_trans_per_iter, render=render)
        # average (undiscounted) reward
        print('avg_rwd', np.sum(paths["rwd"]) / paths["nb_paths"])

        stats.episode_rewards[it] = np.sum(paths["rwd"]) / paths["nb_paths"]
        stats.episode_lengths[it] = it

        # update the v-function
        for epoch in range(epochs_per_iter):
            # compute the generalized td_error and v_targets
            gen_adv, v_values = get_gen_adv(paths, get_v, discount, lam_trace)
            v_targets = v_values + gen_adv  # generalized Bellmann operator

            # log Bellman error if first epoch
            # if epoch == 0:
            #     print('v-function: loss before updating is ', ppo.evaluate_v(paths["obs"], v_targets))
            for batch_idx in next_batch_idx(batch_size, len(v_targets)):
                ppo.train_v(paths["obs"][batch_idx], v_targets[batch_idx])
        # print('v-function: loss after updating is ', ppo.evaluate_v(paths["obs"], v_targets))

        # udpate policy
        gen_adv, _ = get_gen_adv(paths, get_v, discount, lam_trace)
        # print('advantages: std {0:.3f} mean {1:.3f} min {2:.3f} max {3:.3f}'.format(np.std(gen_adv), np.mean(gen_adv), np.min(gen_adv), np.max(gen_adv)))
        gen_adv = gen_adv / np.std(gen_adv)
        log_act_probas = policy.get_log_proba(paths["obs"], paths["act"])
        # print('entropy: before update', session.run(policy.entropy))
        # print('policy: objective before updating ', ppo.evaluate_pol(paths["obs"], paths["act"], old_log_probas=log_act_probas, advantages=gen_adv))
        for epoch in range(epochs_per_iter):
            for batch_idx in next_batch_idx(batch_size, len(gen_adv)):
                ppo.train_pol(paths["obs"][batch_idx], paths["act"][batch_idx], old_log_probas=log_act_probas[batch_idx], advantages=gen_adv[batch_idx])
        # print('policy: objective after updating ', ppo.evaluate_pol(paths["obs"], paths["act"], old_log_probas=log_act_probas, advantages=gen_adv))
        # print('entropy: after update ', session.run(policy.entropy))
        # log_act_probas_new = policy.get_log_proba(paths["obs"], paths["act"])
        # diff = np.exp(log_act_probas - log_act_probas_new)
        # print('action ratio: min {0:.3f} mean {1:.3f} max {2:.3f} std {3:.3f}'.format(np.min(diff), np.mean(diff), np.max(diff), np.std(diff)))
    plotting.plot_episode_stats(stats)

if __name__ == '__main__':
    main()
