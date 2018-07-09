import scipy.optimize

# import numpy as np
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad

import tensorflow as tf
from baselines import logger
import baselines.common.tf_util as U

class EtaOmegaOptimizer(object):
    """
    Finds eta and omega Lagrange multipliers.
    """

    def __init__(self, beta, epsilon, init_eta, init_omega):
        self.init_eta_omega(beta, epsilon, init_eta, init_omega)

    def optimize(self, w_theta, Waa, Wsa, wa, varphis, Kt, prec, is_valid_eta_omega, old_entropy, eta=None):

        # wa = w_beta * \grad_beta \varphi_beta(s) * K^T * Prec

        if False:
            f_dual = self.opt_info['f_dual']
            f_dual_grad = self.opt_info['f_dual_grad']

            # Set BFGS eval function
            def eval_dual(input):
                param_eta = input[0]
                param_omega = input[1]
                val = f_dual(*([varphis, Kt, prec, Waa, Wsa, wa] + [param_eta, param_omega, old_entropy]))
                return val.astype(np.float64)

            # Set BFGS gradient eval function
            def eval_dual_grad(input):
                param_eta = input[0]
                param_omega = input[1]
                grad = f_dual_grad(*([varphis, Kt, prec, Waa, Wsa, wa] + [param_eta, param_omega, old_entropy]))
                return np.asarray(grad)

        if eta is not None:
            param_eta = eta
        else:
            param_eta = self.param_eta

        if self.beta == 0:
            beta = 0
        else:
            beta = old_entropy - self.beta

        # eta_before = param_eta
        # omega_before = self.param_omega
        # dual_before = eval_dual([eta_before, omega_before])
        # dual_grad_before = eval_dual_grad([eta_before, omega_before])

        x0 = [param_eta, self.param_omega]

        # TEST
        # small = 0.000000001
        # f1 = [self.param_eta - small, self.param_omega]
        # f2 = [self.param_eta + small, self.param_omega]
        # fd = (eval_dual(f1) - eval_dual(f2)) / (2 * small)
        #
        # duals = self.opt_info["f_duals"](*([varphis, Kt, prec, Waa, Wsa, wa] + [eta_before, omega_before, old_entropy]))
        # logger.log("Theano eta/omega: " + str(eta_before) + "/" + str(omega_before) + ": " + str(dual_before) +
        #            ", " + str(duals) + ", grad: " + str(eval_dual_grad(x0)) + ", fd: " + str(fd))
        # # END TEST

        # Create dual function
        def eval_dual(input):
            param_eta = input[0]
            param_omega = input[1]

            # ha(s): eta * (\varphi(s)^T * K^T * \Sigma^{-1} + W_{sa}) + wa(s))
            ha = np.dot(varphis, param_eta * np.dot(Kt, prec) + Wsa) + wa

            # hss(s): eta * (\varphi(s)^T * K^T * \Sigma^{-1} * K * \varphi(s))
            varphisKt = np.dot(varphis, Kt)
            hss = param_eta * np.sum(np.dot(varphisKt, prec) * varphisKt, axis=1)

            Haa = param_eta * prec + Waa
            # Haa = 0.5 * (Haa + np.transpose(Haa))
            HaaInv = np.linalg.inv(Haa)

            # The two terms 'term1' and 'term2' which come from normalizers of the
            # 1. Original policy distribution
            # 2. The distribution after completing the square
            sigma = np.linalg.inv(prec)
            term1 = -0.5 * param_eta * np.log(np.linalg.det(2 * np.pi * sigma))
            if self.beta == 0:
                term2 = 0.5 * param_eta * np.log(np.linalg.det(2 * np.pi * param_eta * HaaInv))
            else:
                term2 = 0.5 * (param_eta + param_omega) * np.log(
                    np.linalg.det(2 * np.pi * (param_eta + param_omega) * HaaInv))

            dual = param_eta * self.epsilon - param_omega * beta + \
                   term1 + term2 + np.mean(
                0.5 * (np.sum(np.dot(ha, HaaInv) * ha, axis=1) - hss))

            return dual

        # Automatic gradient of the dual
        eval_dual_grad = grad(eval_dual)

        if True:
            def fx(x):
                eta, omega = x # eta: Lagrange variable of KL constraint, omega: of the entropy constraint
                error_return_val = 1e6, np.array([0., 0.])
                if eta + omega < 0:
                    return error_return_val
                if not is_valid_eta_omega(eta, omega, w_theta):
                    return error_return_val
                return eval_dual(x), eval_dual_grad(x)
        else:
            def fx(x):
                eta, omega = x # eta: Lagrange variable of KL constraint, omega: of the entropy constraint
                error_return_val = 1e6, np.array([0., 0.])
                if eta + omega < 0:
                    return error_return_val
                if not is_valid_eta_omega(eta, omega, w_theta):
                    return error_return_val
                return eval_dual(x), eval_dual_grad(x) # L-BFGS-B expects double floats
                # return np.float64(eval_dual(x)), np.float64(eval_dual_grad(x)) # L-BFGS-B expects double floats

        logger.log('optimizing dual')

        # Make sure valid initial covariance matrices
        while (not is_valid_eta_omega(x0[0], x0[1], w_theta)):
            x0[0] *= 2
            logger.log("Eta increased: " + str(x0[0]))

        if eta is None:
            omega_lower = -100
            if False:
                res = scipy.optimize.minimize(fx, x0, method='L-BFGS-B', jac=True,
                                              bounds=((1e-12, None), (omega_lower, None)), options={'ftol': 1e-12})
            else:
                res = scipy.optimize.minimize(fx, x0, method='SLSQP', jac=True,
                                              bounds=((1e-12, None), (omega_lower, None)), options={'ftol': 1e-12})

            # Make sure that eta > omega
            if res.x[1] < 0 and -res.x[1] > res.x[0]:
                res.x[1] = -res.x[0] + 1e-6
        else:
            # Fixed eta: make sure that eta > omega
            omega_lower = np.max([-(eta - 1e-3) + 1e-6, -100])
            if False:
                res = scipy.optimize.minimize(fx, x0, method='L-BFGS-B', jac=True,
                                              bounds=((eta - 1e-3, eta + 1e-3), (omega_lower, None)),
                                              options={'ftol': 1e-16})
            else:
                res = scipy.optimize.minimize(fx, x0, method='SLSQP', jac=True,
                                              bounds=((eta - 1e-3, eta + 1e-3), (omega_lower, None)), options={'ftol': 1e-16})

        if self.beta == 0:
            res.x[1] = 0

        logger.log("dual optimized, eta: " + str(res.x[0]) + ", omega: " + str(res.x[1]))
        return res.x[0], res.x[1]

        # def f(x, grad):
        #     if grad.size > 0:
        #         grad[:] = eval_dual_grad(x)
        #
        #     return np.float64(eval_dual(x))

        # self.nlopt_opt.set_min_objective(f)
        # # Set parameter boundaries: eta, omega > 0
        # self.nlopt_opt.set_lower_bounds([1e-12, 1e-12])
        #
        # self.nlopt_opt.set_ftol_rel(1e-12)
        # self.nlopt_opt.set_xtol_rel(1e-12)
        # self.nlopt_opt.set_vector_storage(100)

        # try:
        #     x = self.nlopt_opt.optimize([self.param_eta, self.param_omega])
        # except RuntimeError:
        #     entropy = np.mean(self.policy.distribution.entropy_log_probs(samples_data["agent_infos"]))
        #     if entropy < 1e-9:
        #         # ignore error since we already converged and are at the optimal policy
        #         x = [eta_before, omega_before]
        #     else:
        #         print("Error during optimization of the dual...")
        #         raise

        # logger.log('dual optimized')
        #
        # # get optimal values
        # return x[0], x[1]

    def init_eta_omega(self, beta, epsilon, init_eta, init_omega):
        # Here we define the symbolic function for the dual and the gradient

        self.beta = beta
        self.epsilon = epsilon

        # Init dual param values
        self.param_eta = init_eta
        self.param_omega = init_omega

        self.param_eta_non_lin = init_eta
        self.param_omega_non_lin = init_omega

        param_eta = tf.placeholder(dtype=tf.float32, shape=[], name="param_eta")
        param_omega = tf.placeholder(dtype=tf.float32, shape=[], name="param_omega")
        old_entropy = tf.placeholder(dtype=tf.float32, shape=[], name="old_entropy")

        varphis = tf.placeholder(dtype=tf.float32, shape=[None, None], name="varphis")
        Kt = tf.placeholder(dtype=tf.float32, shape=[None, None], name="Kt")
        prec = tf.placeholder(dtype=tf.float32, shape=[None, None], name="prec")
        Waa = tf.placeholder(dtype=tf.float32, shape=[None, None], name="Waa")
        Wsa = tf.placeholder(dtype=tf.float32, shape=[None, None], name="Wsa")
        wa = tf.placeholder(dtype=tf.float32, shape=[None, None], name="wa")

# varphis = ext.new_tensor(
#             'varphis',
#             ndim=2,
#             dtype=theano.config.floatX
#         )
#         Kt = ext.new_tensor(
#             'Kt',
#             ndim=2,
#             dtype=theano.config.floatX
#         )
#         prec = ext.new_tensor(
#             'prec',
#             ndim=2,
#             dtype=theano.config.floatX
#         )
#         Waa = ext.new_tensor(
#             'Waa',
#             ndim=2,
#             dtype=theano.config.floatX
#         )
#         Wsa = ext.new_tensor(
#             'Wsa',
#             ndim=2,
#             dtype=theano.config.floatX
#         )
#         wa = ext.new_tensor(
#             'wa',
#             ndim=2,
#             dtype=theano.config.floatX
#         )

        if self.beta == 0:
            beta = 0
        else:
            beta = old_entropy - self.beta

        # beta = self.printt('beta shape: ', beta)
        # log_action_prob = self.printn('log_action_prob shape: ', log_action_prob)
        # action_prob = self.printn('action_prob shape: ', action_prob)
        # q_values = self.printn('q_values shape: ', q_values)
        # beta = self.printn('beta shape: ', beta)

        # ha(s): eta * (\varphi(s)^T * K^T * \Sigma^{-1} + W_{sa}) + wa(s))
        ha = tf.matmul(varphis, param_eta * tf.matmul(Kt, prec) + Wsa) + wa

        # hss(s): eta * (\varphi(s)^T * K^T * \Sigma^{-1} * K * \varphi(s))
        varphisKt = tf.matmul(varphis, Kt)
        hss = param_eta * tf.reduce_sum(tf.matmul(varphisKt, prec) * varphisKt, axis=1)

        Haa = param_eta * prec + Waa
        # Haa = 0.5 * (Haa + TT.transpose(Haa))
        HaaInv = tf.matrix_inverse(Haa)

        # The two terms 'term1' and 'term2' which come from normalizers of the
        # 1. Original policy distribution
        # 2. The distribution after completing the square
        sigma = tf.matrix_inverse(prec)
        term1 = -0.5 * param_eta * tf.log(tf.matrix_determinant(2 * np.pi * sigma))
        if self.beta == 0:
            term2 = 0.5 * param_eta * tf.log(tf.matrix_determinant(2 * np.pi * param_eta * HaaInv))
        else:
            term2 = 0.5 * (param_eta + param_omega) * tf.log(tf.matrix_determinant(2 * np.pi * (param_eta + param_omega) * HaaInv))

        dual = param_eta * self.epsilon - param_omega * beta + \
            term1 + term2 + tf.reduce_mean(
                0.5 * (tf.reduce_sum(tf.matmul(ha, HaaInv) * ha, axis=1) - hss))

        # Symbolic dual gradient
        dual_grad = tf.gradients(xs=[param_eta, param_omega], ys=dual)

        # Eval functions.
        f_dual = U.function(
            inputs=[varphis, Kt, prec, Waa, Wsa, wa] + [param_eta, param_omega, old_entropy],
            outputs=dual,
#            mode='DebugMode' # TEST
        )

        f_dual_grad = U.function(
            inputs=[varphis, Kt, prec, Waa, Wsa, wa] + [param_eta, param_omega, old_entropy],
            outputs=dual_grad,
 #           mode='DebugMode' # TEST
        )
        #
        # # TEST
        # d0 = param_eta * self.epsilon - param_omega * beta
        # d1 = term1
        # d2 = term2
        # d3 = TT.mean(0.5 * (TT.sum(TT.dot(ha, HaaInv) * ha, axis=1)))
        # d4 = TT.mean(hss)
        # f_duals = ext.compile_function(
        #     inputs=[varphis, Kt, prec, Waa, Wsa, wa] + [param_eta, param_omega, old_entropy],
        #     outputs=[d0, d1, d2, d3, d4]
        # )
        # # END TEST

        self.opt_info = dict(
            f_dual=f_dual,
            f_dual_grad=f_dual_grad,
            # f_duals=f_duals, # TEST
        )

