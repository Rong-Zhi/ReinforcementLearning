import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm

import gym
from baselines.common.distributions import make_pdtype


def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    activ = tf.nn.relu
    scaled_images = unscaled_images / 255.0
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    h3 = activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))
    return h3


class LstmPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_name, m_name, svfname, spiname, ob_space, ac_space, usecnn=False, nlstm=256):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None
        init_std = 1.0
        nenv = 1
        # nbatch = nenv * nsteps

        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)
        self.ob = U.get_placeholder(name=ob_name, dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        M = U.get_placeholder(m_name, tf.float32, [sequence_length])  # mask (done t-1)
        Svf = U.get_placeholder(svfname, tf.float32, [nenv, nlstm * 2])  # states
        Spi = U.get_placeholder(spiname, tf.float32, [nenv, nlstm * 2])  # states

        with tf.variable_scope("vf"):
            if usecnn:
                h = nature_cnn(self.ob)
            else:
                h = self.ob
            # xs = batch_to_seq(h, nenv, nsteps)
            # ms = batch_to_seq(M, nenv, nsteps)
            # h5, vfsnew = lstm(xs, ms, Svf, 'lstmvf', nh=nlstm)
            h5, vfsnew = lstm(h, M, Svf, 'lstmvf', nh=nlstm)
            h5 = seq_to_batch(h5)
            self.vpred = fc(h5, 'value', 1)

        with tf.variable_scope("pol"):

            if usecnn:
                h = nature_cnn(self.ob)
            else:
                h = self.ob
            # xs = batch_to_seq(h, nenv, nsteps)
            # ms = batch_to_seq(M, nenv, nsteps)
            # h5, pisnew = lstm(xs, ms, Spi, 'lstmpi', nh=nlstm)
            h5, pisnew = lstm(h, M, Spi, 'lstmpi', nh=nlstm)
            h5 = seq_to_batch(h5)

            self.action_dim = ac_space.shape[0]
            self.varphi = h5
            self.varphi_dim = 64

            stddev_init = np.ones([1, self.action_dim]) * init_std
            prec_init = 1. / (np.multiply(stddev_init, stddev_init))  # 1 x |a|
            self.prec = tf.get_variable(name="prec", shape=[1, self.action_dim],
                                        initializer=tf.constant_initializer(prec_init))
            kt_init = np.ones([self.varphi_dim, self.action_dim]) * 0.5 / self.varphi_dim
            ktprec_init = kt_init * prec_init
            self.ktprec = tf.get_variable(name="ktprec", shape=[self.varphi_dim, self.action_dim],
                                          initializer=tf.constant_initializer(ktprec_init))
            kt = tf.divide(self.ktprec, self.prec)
            mean = tf.matmul(h5, kt)

            logstd = tf.log(tf.sqrt(1. / self.prec))


            self.prec_get_flat = U.GetFlat([self.prec])
            self.prec_set_from_flat = U.SetFromFlat([self.prec])

            self.ktprec_get_flat = U.GetFlat([self.ktprec])
            self.ktprec_set_from_flat = U.SetFromFlat([self.ktprec])

            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)


        self.pd = pdtype.pdfromflat(pdparam)
        self.M = M
        self.Svf = Svf
        self.Spi = Spi

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, self.ob, M, Spi, Svf], [ac, self.vpred, pisnew, vfsnew])

        # Get all policy parameters
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + '/pol')
        # Remove log-linear parameters ktprec and prec to get only non-linear parameters
        del vars[-1]
        del vars[-1]
        beta_params = vars

        # Flat w_beta
        beta_len = np.sum([np.prod(p.get_shape().as_list()) for p in beta_params])
        w_beta_var = tf.placeholder(dtype=tf.float32, shape=[beta_len])

        # Unflatten w_beta
        beta_shapes = list(map(tf.shape, beta_params))
        w_beta_unflat_var = self.unflatten_tensor_variables(w_beta_var, beta_shapes)

        # w_beta^T * \grad_beta \varphi(s)^T
        v = tf.placeholder(dtype=self.varphi.dtype, shape=self.varphi.get_shape(), name="v_in_Rop")
        features_beta = self.alternative_Rop(self.varphi, beta_params, w_beta_unflat_var, v)

        self.features_beta = U.function([self.ob, w_beta_var, v], features_beta)

    def act(self, stochastic, ob, mask, pistate, vfstate):
        ac1, vpred1, pisnew, vfsnew =  self._act(stochastic, ob[None], mask, pistate, vfstate)
        return ac1[0], vpred1[0], pisnew, vfsnew

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

    def theta_len(self):
        action_dim = self.action_dim
        varphi_dim = self.varphi_dim

        ktprec_len = varphi_dim * action_dim

        prec_len = action_dim

        return (prec_len + ktprec_len)

    def all_to_theta_beta(self, all_params):
        theta_len = self.theta_len()
        theta = all_params[-theta_len:]
        beta = all_params[0:-theta_len]
        return theta, beta

    def theta_beta_to_all(self, theta, beta):
        return np.concatenate([beta, theta])

    def split_w(self, w):
        """
        Split w into w_theta, w_beta
        :param w: [w_beta w_theta]
        :return: w_theta, w_beta
        """
        theta_len = self.theta_len()

        w_beta = w[0:-theta_len]
        w_theta = w[-theta_len:]

        return w_theta, w_beta

    def w2W(self, w_theta):
        """
        Transform w_{theta} to W_aa and W_sa matrices
        :param theta:
        :type theta:
        :return:
        :rtype:
        """
        action_dim = self.action_dim
        varphi_dim = self.varphi_dim

        prec_len = action_dim
        waa = np.reshape(w_theta[0:prec_len], (action_dim,))
        Waa = np.diag(waa)

        Wsa = np.reshape(w_theta[prec_len:], (varphi_dim, action_dim))

        return Waa, Wsa

    def get_wa(self, obs, w_beta):
        """
        Compute wa(s)^T = w_beta^T * \grad_beta \varphi_beta(s)^T * K^T * Sigma^-1
        :return: wa(s)^T
        """
        v0 = np.zeros((obs.shape[0], self.varphi_dim))
        f_beta = self.features_beta(obs, w_beta, v0)[0]
        wa = np.dot(f_beta, self.get_ktprec())

        return wa

    def get_varphis(self, obs):
        # Non-linear neural network outputs
        return tf.get_default_session().run(self.varphi, {self.ob: obs})

    def get_prec_matrix(self):
        return np.diag(self.get_prec().reshape(-1, ))

    def is_policy_valid(self, prec, ktprec):
        if np.any(np.abs(ktprec.reshape(-1, 1)) > 1e12):
            return False

        p = prec

        return np.all(p > 1e-12) and np.all(p < 1e12)

    def is_current_policy_valid(self):
        return self.is_policy_valid(self.get_prec(), self.get_ktprec())

    def is_new_policy_valid(self, eta, omega, w_theta):
        # New policy
        theta_old = self.get_theta()
        theta = (eta * theta_old + w_theta) / (eta + omega)
        prec, ktprec = self.theta2vars(theta)

        return self.is_policy_valid(prec, ktprec)

    def theta2vars(self, theta):
        """
        :param theta:
        :return: [\Sigma^-1, K^T \Sigma^-1],
        """
        action_dim = self.action_dim
        varphi_dim = self.varphi_dim

        prec_len = action_dim
        prec = np.reshape(theta[0:prec_len], (action_dim,))
        ktprec = np.reshape(theta[prec_len:], (varphi_dim, action_dim))

        return (prec, ktprec)

    def get_ktprec(self):
        """
        :return: K^T \Sigma^-1
        """
        return tf.get_default_session().run(self.ktprec)

    def get_prec(self):
        return tf.get_default_session().run(self.prec)

    def get_sigma(self):
        return np.diag(1 / self.get_prec().reshape(-1, ))

    def get_kt(self):
        return np.dot(self.get_ktprec(), self.get_sigma())

    def get_theta(self):
        """
        :return: \theta
        """
        theta = np.concatenate((self.get_prec().reshape(-1, ), self.get_ktprec().reshape(-1, )))
        return theta

    def alternative_Rop(self, f, x, u, v):
        """Alternative implementation of the Rop operation in Theano.
        Please, see
        https://j-towns.github.io/2017/06/12/A-new-trick.html
        https://github.com/renmengye/tensorflow-forward-ad/issues/2
        for an explanation.
        The default value for 'v' should not influence the end result since 'v' is eliminated but
        is needed in some cases to prevent the graph compiler from complaining.
        """
        # v = tf.placeholder_with_default(input=v0, dtype=f.dtype, shape=f.get_shape(), name="v_in_Rop")  # dummy variable
        g = tf.gradients(f, x, grad_ys=v)
        return tf.gradients(g, v, grad_ys=u)

    def unflatten_tensor_variables(self, flatarr, shapes):
        arrs = []
        n = 0
        for shape in shapes:
            size = tf.reduce_prod(shape)
            arr = tf.reshape(flatarr[n:n + size], shape)
            arrs.append(arr)
            n += size
        return arrs
