from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as utils
import tensorflow as tf
import numpy as np
from baselines.common.distributions import make_pdtype
import gym


class CompatibleMlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, init_std=1.0, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        self.varphi_dim = hid_size

        self.ob = utils.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        # self.ob = tf.placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=utils.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=utils.normc_initializer(1.0))[:, 0]

        with tf.variable_scope('pol'):
            last_out = obz
            # Create 'num_hid_layers' hidden layers
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=utils.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                self.action_dim = ac_space.shape[0]

                # mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                # logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                # pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)

                self.dist_diagonal = True
                self.varphi = last_out
                self.varphi_dim = hid_size

                if self.dist_diagonal:
                    stddev_init = np.ones([1, self.action_dim]) * init_std
                    prec_init = 1. / (np.multiply(stddev_init, stddev_init))  # 1 x |a|
                    self.prec = tf.get_variable(name="prec", shape=[1, self.action_dim],
                                                initializer=tf.constant_initializer(prec_init))
                    kt_init = np.ones([self.varphi_dim, self.action_dim]) * 0.5 / self.varphi_dim
                    ktprec_init = kt_init * prec_init
                    self.ktprec = tf.get_variable(name="ktprec", shape=[self.varphi_dim, self.action_dim],
                                                  initializer=tf.constant_initializer(ktprec_init))
                    kt = tf.divide(self.ktprec, self.prec)
                    mean = tf.matmul(last_out, kt)

                    logstd = tf.log(tf.sqrt(1. / self.prec))
                else:
                    # Not implemented yet
                    raise NotImplementedError

                self.prec_get_flat = utils.GetFlat([self.prec])
                self.prec_set_from_flat = utils.SetFromFlat([self.prec])

                self.ktprec_get_flat = utils.GetFlat([self.ktprec])
                self.ktprec_set_from_flat = utils.SetFromFlat([self.ktprec])

                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=utils.normc_initializer(0.01))

        self.scope = tf.get_variable_scope().name

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = utils.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = utils.function([stochastic, self.ob], [ac, self.vpred])

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

        self.features_beta = utils.function([self.ob, w_beta_var, v], features_beta)

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

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
        if self.dist_diagonal:
            prec_len = action_dim
        else:
            prec_len = action_dim * action_dim

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

        if self.dist_diagonal:
            prec_len = action_dim
            waa = np.reshape(w_theta[0:prec_len], (action_dim,))
            Waa = np.diag(waa)
        else:
            prec_len = action_dim * action_dim
            Waa = np.reshape(w_theta[0:prec_len], (action_dim,action_dim))

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
        if self.dist_diagonal:
            return np.diag(self.get_prec().reshape(-1,))
        else:
            return self.get_prec()

    def is_policy_valid(self, prec, ktprec):
        if np.any(np.abs(ktprec.reshape(-1,1)) > 1e12):
            return False

        if self.dist_diagonal:
            p = prec
        else:
            p = np.linalg.eigvals(prec)

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

        if self.dist_diagonal:
            prec_len = action_dim
            prec = np.reshape(theta[0:prec_len], (action_dim,))
            ktprec = np.reshape(theta[prec_len:], (varphi_dim, action_dim))
        else:
            prec_len = action_dim * action_dim
            prec = np.reshape(theta[0:prec_len],
                              (action_dim, action_dim))
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
        if self.dist_diagonal:
            return np.diag(1 / self.get_prec().reshape(-1, ))
        else:
            return np.linalg.inv(self.get_prec())

    def get_kt(self):
        return np.dot(self.get_ktprec(), self.get_sigma())

    def get_theta(self):
        """
        :return: \theta
        """
        theta = np.concatenate((self.get_prec().reshape(-1,), self.get_ktprec().reshape(-1,)))
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

    # #
    # # The following two functions are needed to compute a Jacobian in tensorflow
    # #
    # def map(self, f, x, dtype=None, parallel_iterations=10):
    #     '''
    #     Apply f to each of the elements in x using the specified number of parallel iterations.
    #
    #     Important points:
    #     1. By "elements in x", we mean that we will be applying f to x[0],...x[tf.shape(x)[0]-1].
    #     2. The output size of f(x[i]) can be arbitrary. However, if the dtype of that output
    #        is different than the dtype of x, then you need to specify that as an additional argument.
    #     '''
    #     if dtype is None:
    #         dtype = x.dtype
    #
    #     n = tf.shape(x)[0]
    #     loop_vars = [
    #         tf.constant(0, n.dtype),
    #         tf.TensorArray(dtype, size=n),
    #     ]
    #     _, fx = tf.while_loop(
    #         lambda j, _: j < n,
    #         lambda j, result: (j + 1, result.write(j, f(x[j]))),
    #         loop_vars,
    #         parallel_iterations=parallel_iterations
    #     )
    #     return fx.stack()
    #
    # def jacobian(self, fx, x, parallel_iterations=10):
    #     '''
    #     Given a tensor fx, which is a function of x, vectorize fx (via tf.reshape(fx, [-1])),
    #     and then compute the jacobian of each entry of fx with respect to x.
    #     Specifically, if x has shape (m,n,...,p), and fx has L entries (tf.size(fx)=L), then
    #     the output will be (L,m,n,...,p), where output[i] will be (m,n,...,p), with each entry denoting the
    #     gradient of output[i] wrt the corresponding element of x.
    #     '''
    #     return self.map(lambda fxi: tf.gradients(fxi, x)[0],
    #                     tf.reshape(fx, [-1]),
    #                     dtype=x.dtype,
    #                     parallel_iterations=parallel_iterations)
    #
    #
