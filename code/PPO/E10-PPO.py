import numpy as np
import tensorflow as tf
from lib import plotting
import matplotlib
import matplotlib.pyplot as plt
import gym

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
        self.act_space = env.action_space.shape[0]
        self.state_space = env.observation_space.shape[0]
        self.init_sigma = 1.5
        activ = tf.nn.tanh
        initial = tf.glorot_uniform_initializer()

        with tf.variable_scope("policy"):
            # placeholder
            self.states_p = tf.placeholder(tf.float32, shape=[None, self.state_space], name='states_p')
            self.test_action_p = tf.placeholder(tf.float32, shape=[None, self.act_space], name='test_action')

            # layers
            self.l1_p = tf.layers.dense(inputs=self.states_p, units=64, activation=activ,
                                        kernel_initializer=initial, name='pl1')
            self.l2_p = tf.layers.dense(self.l1_p, 64, activation=activ,
                                        kernel_initializer=initial,name='pl2')

            # outputs
            self.mu = tf.layers.dense(inputs=self.l2_p, units=self.act_space, activation=None,
                                      kernel_initializer=initial, name='mu')

            # self.sigma = tf.layers.dense(self.l2, self.act_space, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), name='sigma')
            # self.sigma = tf.nn.softplus(self.sigma) + 1e-5


            self.sigma = tf.Variable(np.log(self.init_sigma) * tf.ones([1, self.act_space]))
            self.sigma = tf.exp(self.sigma)

            # check the exploration situation
            self.entropy = tf.reduce_sum(self.sigma) + self.act_space * np.log(2 * np.pi * np.e) / 2

            # action probability and action
            self.new_policy_dist_p = tf.contrib.distributions.MultivariateNormalDiag(self.mu, self.sigma)
            self.action_p = self.new_policy_dist_p.sample()
            # self.action_p = tf.clip_by_value(self.action_p, self.act_low, self.act_high)

            self.new_log_prob_p = tf.expand_dims(self.new_policy_dist_p.log_prob(self.test_action_p), axis=-1)


    def predict_action(self, state_p):
        return np.squeeze(self.sess.run(self.action_p, {self.states_p: np.asmatrix(state_p)}), axis=0)

    def predict_log_dist(self, state_p, action_p):
        return self.sess.run(self.new_log_prob_p, {self.states_p:state_p, self.test_action_p:action_p})



class Value_net(object):
    """
    Critic net: build neural network with 2 hidden layers,
                each layer contains 64 neurons,
                learning rate = 0.0002
                optimizer -- Adamoptimizer
    """
    def __init__(self, env, sess):
        self.sess = sess
        self.act_space = env.action_space.shape[0]
        self.state_space = env.observation_space.shape[0]

        activ = tf.nn.tanh
        initial = tf.glorot_uniform_initializer()

        with tf.variable_scope("value"):

            # placeholder
            self.states_v = tf.placeholder(tf.float32, shape=[None, self.state_space], name='states_v')

            # layers
            self.l1_v = tf.layers.dense(inputs=self.states_v, units=64, activation=activ,
                                        kernel_initializer=initial, name='vl1')
            self.l2_v = tf.layers.dense(self.l1_v, 64, activation=activ,
                                        kernel_initializer=initial, name='vl2')

            # output
            self.output_v = tf.layers.dense(inputs=self.l2_v, units=1, activation=None,
                                            kernel_initializer=initial, name='output')
            # self.value_estimate = tf.squeeze(self.output_v, name='squeeze_output')

    def predict(self, state_v):
        return self.sess.run(self.output_v, {self.states_v : state_v})


def rollout(env, get_policy):
    state = env.reset()
    done = False
    while not done:
        action = get_policy(state_p=state)
        next_state, reward, done, _ = env.step(
            np.minimum(np.maximum(action, env.action_space.low), env.action_space.high))
        yield state, action, reward, done
        state = next_state


def rollouts(env, get_policy, timestep):
    keys = ['states', 'action', 'reward', 'done']
    path = {}
    for k in keys:
        path[k] = []
    nb_paths = 0
    while len(path["reward"]) < timestep:
        for trans in rollout(env=env, get_policy=get_policy):
            for key, val in zip(keys, trans):
                path[key].append(val)
        nb_paths += 1
    for key in keys:
        path[key] = np.asarray(path[key])
        if path[key].ndim == 1:
            path[key] = np.expand_dims(path[key], axis=-1)
    path['nb_paths'] = nb_paths
    return path


def compute_advantage(get_value, paths, discount_factor):
    values = get_value(state_v=paths['states'])
    gen_adv = np.empty_like(values)
    for rev_k, v in enumerate(reversed(values)):
        k = len(values) - rev_k - 1
        if paths['done'][k]:  # this is a new path. always true for rev_k == 0
            gen_adv[k] = paths['reward'][k] - values[k]
        else:
            gen_adv[k] = paths['reward'][k] + discount_factor * values[k + 1] - \
                         values[k] + discount_factor * 0.95 * gen_adv[k + 1]
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
        self.vlr = 4e-4
        self.plr = 5e-4
        self.epsilon = 0.2
        self.v_coef = 0.5

        with tf.variable_scope('ppo'):
            # loss for value estimator
            self.target_v = tf.placeholder(tf.float32, shape=[None, 1], name='target_v')

            self.loss_v = tf.losses.mean_squared_error(self.value.output_v, self.target_v)
            self.optimizer_v = tf.train.AdamOptimizer(learning_rate=self.vlr)
            self.train_op_v = self.optimizer_v.minimize(self.loss_v)

            # loss for policy estimator
            self.advantage_p = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')
            self.old_log_prob_p = tf.placeholder(tf.float32, shape=[None, 1], name='old_log_prob')

            ratio = tf.exp(self.policy.new_log_prob_p - self.old_log_prob_p)
            clip_p = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)

            # loss and optimizer
            self.surrogate_loss = -tf.reduce_mean(tf.minimum(tf.multiply(ratio, self.advantage_p),
                                                             tf.multiply(clip_p, self.advantage_p)),
                                                  name='surrogate_loss')

            self.optimizer_p = tf.train.AdamOptimizer(learning_rate=self.plr)
            self.train_op_p = self.optimizer_p.minimize(self.surrogate_loss)

            self.loss_all = self.surrogate_loss + self.loss_v * self.v_coef
            self.train_optimizer_all = tf.train.AdamOptimizer(self.plr).minimize(self.loss_all)



    def update_v(self, state_v, target_v):
        feed_dict = {self.value.states_v: state_v, self.target_v: target_v}
        _, loss_v = self.sess.run([self.train_op_v, self.loss_v],
                                        feed_dict=feed_dict)
        return loss_v

    def update_p(self, state_p, advantage_p, test_action_p, old_prob_p):

        feed_dict = {self.policy.states_p:state_p, self.advantage_p: advantage_p,
                     self.policy.test_action_p:test_action_p,
                     self.old_log_prob_p:old_prob_p}
        _ ,loss_p = self.sess.run([self.train_op_p, self.surrogate_loss],
                                        feed_dict=feed_dict)
        return loss_p

    def update_all(self, state, target_v, advantage_p, test_action_p, old_prob_p):

        feed_dict = {self.value.states_v:state, self.target_v: target_v,
                     self.policy.states_p:state, self.advantage_p: advantage_p,
                     self.policy.test_action_p:test_action_p,
                     self.old_log_prob_p:old_prob_p}

        self.sess.run([self.train_optimizer_all], feed_dict=feed_dict)

def main():
    env = gym.make("MountainCarContinuous-v0")
    # env = gym.make('Pendulum-v0')

    seed = 1234
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)

    sess = tf.Session()

    policy_estimator = Policy_net(env=env, sess=sess)
    value_estimator = Value_net(env=env, sess=sess)
    ppo = PPO(sess=sess, policy_estimator=policy_estimator,
              value_estimator=value_estimator)

    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter('./log', sess.graph)

    discount_factor = 0.99
    num_iteration = 100
    timestep = 3200
    batchsize = 64
    epoch_per_iter = 15

    result = {'Reward':[], 'Entropy':[]}

    for i_iteration in range(num_iteration):

        paths = rollouts(env=env, get_policy=policy_estimator.predict_action, timestep=timestep)
        print('Training {0}/{1}, reward:{2} \n'.format(i_iteration, num_iteration, np.sum(paths['reward'])/paths['nb_paths']))

        result['Entropy'].append(sess.run(policy_estimator.entropy))
        result['Reward'].append(np.sum(paths['reward']) / paths['nb_paths'])

        # Actor-Critic style

        # update value estimator
        # for epoch in range(epoch_per_iter):
        #     advantage, target = compute_advantage(get_value=value_estimator.predict,
        #                                           paths=paths,
        #                                           discount_factor=discount_factor)
        #     for idx in next_batch_idx(batchsize, len(target)):
        #         lossv = ppo.update_v(state_v=paths['states'][idx],
        #                              target_v=target[idx])
        #
        # # update policy & value estimator
        # advantage, _ = compute_advantage(get_value=value_estimator.predict,
        #                                       paths=paths,
        #                                       discount_factor=discount_factor)
        # # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # advantage = advantage/np.std(advantage)
        # old_log_prob = policy_estimator.predict_log_dist(state_p=paths['states'],
        #                                                  action_p=paths['action'])
        # for epoch in range(epoch_per_iter):
        #     for idx in next_batch_idx(batchsize,len(advantage)):
        #         lossp = ppo.update_p(state_p=paths['states'][idx],
        #                              advantage_p=advantage[idx],
        #                              test_action_p=paths['action'][idx],
        #                              old_prob_p=old_log_prob[idx])


        # Update using one loss
        advantage, target = compute_advantage(get_value=value_estimator.predict,
                                              paths=paths,
                                              discount_factor=discount_factor)

        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        advantage = advantage / np.std(advantage)
        old_log_prob = policy_estimator.predict_log_dist(state_p=paths['states'],
                                                         action_p=paths['action'])
        for epoch in range(epoch_per_iter):
            for idx in next_batch_idx(batchsize,len(advantage)):
                ppo.update_all(state=paths['states'][idx],
                                       target_v=target[idx],
                                     advantage_p=advantage[idx],
                                     test_action_p=paths['action'][idx],
                                     old_prob_p=old_log_prob[idx])


        # print('Loss: Value estimator:{0}, Policy estimator:{1}'.format(lossv, lossp))
        print('Entropy of policy:{0}'.format(sess.run(policy_estimator.entropy)))




    plt.figure(1)
    plt.plot(result['Reward'])
    plt.ylabel('Reward')
    plt.title('Reward')
    plt.figure(2)
    plt.plot(result['Entropy'])
    plt.ylabel('Entropy')
    plt.title('Entropy')
    plt.show()


    # state = env.reset()
    # while True:
    #     env.render()
    #     action = policy_estimator.predict_action(state_p=state)
    #     next_state, reward, done, _ = env.step(action)
    #     # if done:
    #     #     break
    #     state = next_state


if __name__ == '__main__':
    main()
