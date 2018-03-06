
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import gym

from scipy.stats import entropy

from rllab.envs.normalized_env import normalize
from rllab.envs.pomdp.rock_sample_env import RockSampleEnv
from rllab.envs.history_env import HistoryEnv


matplotlib.style.use('ggplot')


class Policy_net(object):
    """
    Actor net: build neural network with 2 hidden layers,
                each layer contains 64 neurons,
                output of network is mu and sigma of gaussian distribution,

                learning_rate = 0.0001
                epsilon = 0.2
    """
    def __init__(self, env, sess):

        self.sess = sess
        self.act_space = env.action_space.n
        self.state_space = env.observation_space.shape[0]
        activ = tf.nn.tanh
        initial = tf.glorot_uniform_initializer()

        with tf.variable_scope("policy"):
            # placeholder
            self.states_p = tf.placeholder(tf.float32, shape=[None, self.state_space], name='states_p')
            self.test_action_p = tf.placeholder(tf.int32, shape=[None], name='test_action')

            # layers
            self.l1_p = tf.layers.dense(inputs=self.states_p, units=32, activation=activ,
                                        kernel_initializer=initial, name='pl1')
            self.l2_p = tf.layers.dense(self.l1_p, 32, activation=activ,
                                        kernel_initializer=initial,name='pl2')

            # output action probability
            self.action_p = tf.layers.dense(inputs=self.l2_p, units=self.act_space,
                                          activation=tf.nn.softmax,
                                          kernel_initializer=initial, name='output')
            # entropy
            self.entropy = tf.reduce_mean(tf.reduce_sum(-self.action_p * tf.log(self.action_p), axis=1))

            # action will be taken
            self.action = tf.multinomial(tf.log(self.action_p), 1)

            # self.new_log_prob_p = lambda s: self.sess.run(self.action_p, {self.states_p: s})[:, self.test_action_p]
            self.new_log_prob_p = self.get_idx_value(self.action_p, self.test_action_p)


    def get_idx_value(self, x, idx):
        shape = tf.cast(tf.shape(x), tf.int32)
        inds = tf.range(0, shape[0])
        x_flat = tf.reshape(x,[-1])
        value = tf.gather(x_flat, inds*shape[1] + idx)
        return value

    def predict_entropy(self, state_p):
        return self.sess.run(self.entropy, {self.states_p:np.asmatrix(state_p)})

    def predict_action(self, state_p):
        return np.squeeze(self.sess.run(self.action, {self.states_p: np.asmatrix(state_p)}), axis=0)

    def predict_log_dist(self, state_p, test_action_p):
        x = self.sess.run(self.new_log_prob_p, {self.states_p:state_p, self.test_action_p:test_action_p})
        return x



class Value_net(object):
    """
    Critic net: build neural network with 2 hidden layers,
                each layer contains 64 neurons,
                learning rate = 0.0002
                optimizer -- Adamoptimizer
    """
    def __init__(self, env, sess):
        self.sess = sess
        self.act_space = env.action_space.n
        self.state_space = env.observation_space.shape[0]

        activ = tf.nn.tanh
        initial = tf.glorot_uniform_initializer()

        with tf.variable_scope("value"):

            # placeholder
            self.states_v = tf.placeholder(tf.float32, shape=[None, self.state_space], name='states_v')

            # layers
            self.l1_v = tf.layers.dense(inputs=self.states_v, units=32, activation=activ,
                                        kernel_initializer=initial, name='vl1')
            self.l2_v = tf.layers.dense(self.l1_v, 32, activation=activ,
                                        kernel_initializer=initial, name='vl2')

            # output
            self.output_v = tf.layers.dense(inputs=self.l2_v, units=1, activation=None,
                                            kernel_initializer=initial, name='output')


    def predict(self, state_v):
        return self.sess.run(self.output_v, {self.states_v : state_v})


def rollout(env, get_policy):
    state = env.reset()
    done = False
    while not done:
        action = get_policy(state_p=state)
        next_state, reward, done, _ = env.step(action[0])
        yield state, action, reward, done
        state = next_state


def rollouts(env, get_policy, timestep, df):
    keys = ['states', 'action', 'reward', 'done', '']
    path = {}
    for k in keys:
        path[k] = []
    nb_paths = 0
    discount_reward = []
    while len(path["reward"]) < timestep:
        drwd = []
        for trans in rollout(env=env, get_policy=get_policy):
            for key, val in zip(keys, trans):
                path[key].append(val)
            drwd.append(trans[2])
        discount_reward.append(np.sum([rwd * df**n for n,rwd in enumerate(drwd)]))
        nb_paths += 1
    for key in keys:
        path[key] = np.asarray(path[key])
        if path[key].ndim == 1:
            path[key] = np.expand_dims(path[key], axis=-1)
    path['nb_paths'] = nb_paths
    path['dis_avg_rwd'] = np.mean(discount_reward)
    return path


def compute_advantage(get_value, paths, discount_factor):
    lam = 0.95
    values = get_value(state_v=paths['states'])
    gen_adv = np.empty_like(values)
    for rev_k, v in enumerate(reversed(values)):
        k = len(values) - rev_k - 1
        if paths['done'][k]:  # this is a new path. always true for rev_k == 0
            gen_adv[k] = paths['reward'][k] - values[k]
        else:
            gen_adv[k] = paths['reward'][k] + discount_factor * values[k + 1] - \
                         values[k] + discount_factor * lam * gen_adv[k + 1]
    return gen_adv, gen_adv + values


def next_batch_idx(batch_size, data_set_size):
    batch_idx_list = np.random.choice(data_set_size, data_set_size, replace=False)
    for batch_start in range(0, data_set_size, batch_size):
        yield batch_idx_list[batch_start: min(batch_start+batch_size, data_set_size)]


class PPO:
    def __init__(self, sess, policy_estimator, value_estimator):
        self.sess = sess
        self.policy = policy_estimator
        self.value = value_estimator
        self.lr = 2.5e-4
        self.epsilon = 0.2
        self.v_coef = 0.5
        self.ent_coef = 0.0

        with tf.variable_scope('ppo'):
            # loss for value estimator
            self.target_v = tf.placeholder(tf.float32, shape=[None, 1], name='target_v')

            self.loss_v = tf.losses.mean_squared_error(self.value.output_v, self.target_v)

            # loss for policy estimator
            self.advantage_p = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')
            self.old_log_prob_p = tf.placeholder(tf.float32, shape=[None], name='old_log_prob')

            ratio = tf.exp(self.policy.new_log_prob_p - self.old_log_prob_p)

            clip_p = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)

            # loss and optimizer
            self.surrogate_loss = -tf.reduce_mean(tf.minimum(tf.multiply(ratio, self.advantage_p),
                                                             tf.multiply(clip_p, self.advantage_p)),
                                                  name='surrogate_loss')

            self.loss_all = self.surrogate_loss + self.v_coef * self.loss_v - self.ent_coef * self.policy.entropy
            self.train_all = tf.train.AdamOptimizer(self.lr).minimize(self.loss_all)



    def update_all(self, state, target_v, advantage_p, test_action_p, old_prob_p):
        feed_dict = {self.value.states_v:state, self.target_v:target_v,
                     self.policy.states_p: state, self.advantage_p: advantage_p,
                     self.policy.test_action_p: test_action_p,
                     self.old_log_prob_p: old_prob_p}
        _, loss_all, sloss, vloss, eloss = self.sess.run([self.train_all, self.loss_all,
                                                          self.surrogate_loss, self.loss_v, self.policy.entropy],
                                                         feed_dict=feed_dict)
        return loss_all, sloss, vloss, eloss


# def main():


env = normalize(HistoryEnv(
    RockSampleEnv(map_name="5x7", observation_type="field_vision_full_pos", observation_noise=True)
    , n_steps=15), scale_reward=1)

# env = gym.envs.make('CartPole-v0')


# seed = 1
# np.random.seed(seed)
# tf.set_random_seed(seed)
# env.seed(seed)

sess = tf.Session()

policy_estimator = Policy_net(env=env, sess=sess)
value_estimator = Value_net(env=env, sess=sess)
ppo = PPO(sess=sess, policy_estimator=policy_estimator,
          value_estimator=value_estimator)

sess.run(tf.global_variables_initializer())

# tf.summary.FileWriter('./log', sess.graph)

discount_factor = 0.95
num_iteration = 600
timestep = 5000
batchsize = 64
epoch_per_iter = 35
average_time = 1


all_result = []
for i in range(average_time):
    result = {'Reward': [], 'Entropy': []}
    for i_iteration in range(num_iteration):

        paths = rollouts(env=env, get_policy=policy_estimator.predict_action, timestep=timestep, df=discount_factor)

        ent = policy_estimator.predict_entropy(state_p=paths['states'])
        result['Entropy'].append(ent)

        # update policy & value estimator
        advantage, target = compute_advantage(get_value=value_estimator.predict,
                                              paths=paths,
                                              discount_factor=discount_factor)
        # calculate average return
        # result['Reward'].append(np.sum(target) / paths['nb_paths'])

        # print result
        # print('Training {0}/{1}, reward:{2} \n'.format(i_iteration, num_iteration,
        #                                                np.sum(paths['reward'])/paths['nb_paths']))

        # average dicount reward
        result['Reward'].append(paths['dis_avg_rwd'])

        print('Training {0}/{1}, reward:{2} \n'.format(i_iteration, num_iteration, paths['dis_avg_rwd']))
        print('Entropy of policy:{0}'.format(ent))

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        old_log_prob = policy_estimator.predict_log_dist(state_p=paths['states'],
                                                         test_action_p=np.squeeze(paths['action']))

        for epoch in range(epoch_per_iter):
            for idx in next_batch_idx(batchsize,len(advantage)):
                loss_all, sloss, vloss, eloss = ppo.update_all(state=paths['states'][idx],
                                       target_v=target[idx],
                                     advantage_p=advantage[idx],
                                     test_action_p=np.squeeze(paths['action'][idx]),
                                     old_prob_p=old_log_prob[idx])



        print('Loss: {0}'.format(loss_all))
        print('Surrogate loss:{0}, Value loss:{1}, entropy loss:{2}'.format(sloss, vloss, eloss))
    all_result.append(result)

average_reward = np.mean([res['Reward'] for res in all_result], axis=0)
# reward_std = np.std([res['Reward'] for res in all_result], axis=0)
average_entropy = np.mean([res['Entropy'] for res in all_result], axis=0)
# entropy_std = np.std([res['Entropy'] for res in all_result], axis=0)


rewards_smoothed = pd.Series(average_reward).rolling(5, min_periods=5).mean()
entropy_smoothed = pd.Series(average_entropy).rolling(5, min_periods=5).mean()


# p_range = np.linspace(0, num_iteration-1)
plt.figure(1)
plt.plot(rewards_smoothed)
# plt.fill_between(average_reward - reward_std,
#                  average_reward + reward_std, alpha=0.2,
#                  color="coral")
plt.ylabel('Average Reward')
plt.title('Average Reward over 10 times')

plt.figure(2)
plt.plot(entropy_smoothed)
# plt.fill_between(average_entropy - entropy_std,
#                  average_entropy + entropy_std, alpha=0.2,
#                  color='coral')
plt.ylabel('Average Entropy')
plt.title('Average Entropy over 10 times')

plt.show()



    # state = env.reset()
    # while True:
    #     env.render()
    #     action = policy_estimator.predict_action(state_p=state)
    #     next_state, reward, done, _ = env.step(action)
    #     # if done:
    #     #     break
    #     state = next_state


# if __name__ == '__main__':
#     main()


