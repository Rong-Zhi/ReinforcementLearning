from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager

from baselines.copos.eta_omega_dual import EtaOmegaOptimizer


def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def eta_search(w_theta, w_beta, eta, omega, allmean, compute_losses, get_flat, set_from_flat, pi, epsilon, args):
    """
    Binary search for eta for finding both valid log-linear "theta" and non-linear "beta" parameter values
    :return: new eta
    """

    w_theta = w_theta.reshape(-1,)
    w_beta = w_beta.reshape(-1,)
    all_params = get_flat()
    best_params = all_params
    param_theta, param_beta = pi.all_to_theta_beta(all_params)
    prev_param_theta = np.copy(param_theta)
    prev_param_beta = np.copy(param_beta)
    final_gain = -1e20
    final_constraint_val = float('nan')
    gain_before, kl, *_ = allmean(np.array(compute_losses(*args)))

    min_ratio = 0.1
    max_ratio = 10
    ratio = max_ratio

    for _ in range(10):
        cur_eta = ratio * eta
        cur_param_theta = (cur_eta * prev_param_theta + w_theta) / (cur_eta + omega)
        cur_param_beta = prev_param_beta + w_beta / cur_eta

        thnew = pi.theta_beta_to_all(cur_param_theta, cur_param_beta)
        set_from_flat(thnew)

        # TEST
        if np.min(np.real(np.linalg.eigvals(pi.get_prec_matrix()))) < 0:
            print("Negative definite covariance!")

        if np.min(np.imag(np.linalg.eigvals(pi.get_prec_matrix()))) != 0:
            print("Covariance has imaginary eigenvalues")

        gain, kl, *_ = allmean(np.array(compute_losses(*args)))

        # TEST
        # print(ratio, gain, kl)

        if all((not np.isnan(kl), kl <= epsilon)):
            if all((not np.isnan(gain), gain > final_gain)):
                eta = cur_eta
                final_gain = gain
                final_constraint_val = kl
                best_params = thnew

            max_ratio = ratio
            ratio = 0.5 * (max_ratio + min_ratio)
        else:
            min_ratio = ratio
            ratio = 0.5 * (max_ratio + min_ratio)

    if any((np.isnan(final_gain), np.isnan(final_constraint_val), final_constraint_val >= epsilon)):
        logger.log("eta_search: Line search condition violated. Rejecting the step!")
        if np.isnan(final_gain):
            logger.log("eta_search: Violated because gain is NaN")
        if np.isnan(final_constraint_val):
            logger.log("eta_search: Violated because KL is NaN")
        if final_gain < gain_before:
            logger.log("eta_search: Violated because gain not improving")
        if final_constraint_val >= epsilon:
            logger.log("eta_search: Violated because KL constraint violated")
        set_from_flat(all_params)
    else:
        set_from_flat(best_params)

    logger.log("eta optimization finished, final gain: " + str(final_gain))

    return eta

# def eta_search(w_theta, w_beta, eta, omega, allmean, compute_losses, get_flat, set_from_flat, pi, epsilon, args):
#     """
#     Binary search for eta for finding both valid log-linear "theta" and non-linear "beta" parameter values
#     :return: new eta
#     """
#
#     w_theta = w_theta.reshape(-1,)
#     w_beta = w_beta.reshape(-1,)
#     all_params = get_flat()
#     best_params = all_params
#     param_theta, param_beta = pi.all_to_theta_beta(all_params)
#     prev_param_theta = np.copy(param_theta)
#     prev_param_beta = np.copy(param_beta)
#     final_gain = -1e20
#     final_constraint_val = float('nan')
#     gain_before, kl, *_ = allmean(np.array(compute_losses(*args)))
#
#     backtrack_ratio = 0.8
#     max_backtracks = 20
#
#     gain = gain_before
#     for n_iter, ratio in enumerate(np.concatenate([9.32 * (backtrack_ratio ** np.arange(1, max_backtracks))])):
#         cur_eta = ratio * eta
#         cur_param_theta = (cur_eta * prev_param_theta + w_theta) / (cur_eta + omega)
#         cur_param_beta = prev_param_beta + w_beta / cur_eta
#
#         thnew = pi.theta_beta_to_all(cur_param_theta, cur_param_beta)
#         set_from_flat(thnew)
#
#         # TEST
#         if np.min(np.real(np.linalg.eigvals(pi.get_prec_matrix()))) < 0:
#             print("Negative definite covariance!")
#
#         if np.min(np.imag(np.linalg.eigvals(pi.get_prec_matrix()))) != 0:
#             print("Covariance has imaginary eigenvalues")
#
#         gain, kl, *_ = allmean(np.array(compute_losses(*args)))
#
#         # TEST
#         print(ratio, gain, kl)
#
#         if all((gain > final_gain, kl <= epsilon, not np.isnan(kl), not np.isnan(gain))):
#             eta = cur_eta
#             final_gain = gain
#             final_constraint_val = kl
#             best_params = thnew
#
#     if any((np.isnan(final_gain), np.isnan(final_constraint_val), final_constraint_val >= epsilon)):
#         logger.log("eta_search: Line search condition violated. Rejecting the step!")
#         if np.isnan(final_gain):
#             logger.log("eta_search: Violated because gain is NaN")
#         if np.isnan(final_constraint_val):
#             logger.log("eta_search: Violated because KL is NaN")
#         if final_gain < gain_before:
#             logger.log("eta_search: Violated because gain not improving")
#         if final_constraint_val >= epsilon:
#             logger.log("eta_search: Violated because KL constraint violated")
#         set_from_flat(all_params)
#     else:
#         set_from_flat(best_params)
#
#     logger.log("eta optimization finished, final gain: " + str(final_gain))
#     return eta




def learn(env, policy_fn, *,
          timesteps_per_batch,  # what to train on
          epsilon, beta, cg_iters,
          gamma, lam,  # advantage estimation
          trial, method,
          entcoeff=0.0,
          cg_damping=1e-2,
          vf_stepsize=3e-4,
          vf_iters =3,
          max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
          callback=None,
          TRPO=False
          ):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    old_entropy = oldpi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    # ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    # surrgain = tf.reduce_mean(ratio * atarg)

    surrgain = tf.reduce_mean(pi.pd.logp(ac) * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    all_var_list = [v for v in all_var_list if v.name.split("/")[0].startswith("pi")]
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Initialize eta, omega optimizer
    init_eta = 0.5
    init_omega = 2.0
    eta_omega_optimizer = EtaOmegaOptimizer(beta, epsilon, init_eta, init_omega)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()


            if TRPO:
                #
                # TRPO specific code.
                # Find correct step size using line search
                #
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / epsilon)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > epsilon * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(thbefore)
            else:
                #
                # COPOS specific implementation.
                #

                copos_update_dir = stepdir

                # Split direction into log-linear 'w_theta' and non-linear 'w_beta' parts
                w_theta, w_beta = pi.split_w(copos_update_dir)

                # q_beta(s,a) = \grad_beta \log \pi(a|s) * w_beta
                #             = features_beta(s) * K^T * Prec * a
                # q_beta = self.target.get_q_beta(features_beta, actions)

                Waa, Wsa = pi.w2W(w_theta)
                wa = pi.get_wa(ob, w_beta)

                varphis = pi.get_varphis(ob)

                # Optimize eta and omega
                tmp_ob = np.zeros((1,) + env.observation_space.shape) # We assume that entropy does not depend on the NN
                old_ent = old_entropy.eval({oldpi.ob: tmp_ob})[0]
                eta, omega = eta_omega_optimizer.optimize(w_theta, Waa, Wsa, wa, varphis, pi.get_kt(),
                                                          pi.get_prec_matrix(), pi.is_new_policy_valid, old_ent)
                logger.log("Initial eta: " + str(eta) + " and omega: " + str(omega))

                current_theta_beta = get_flat()
                prev_theta, prev_beta = pi.all_to_theta_beta(current_theta_beta)

                for i in range(2):
                    # Do a line search for both theta and beta parameters by adjusting only eta
                    eta = eta_search(w_theta, w_beta, eta, omega, allmean, compute_losses, get_flat, set_from_flat, pi,
                                     epsilon, args)
                    logger.log("Updated eta, eta: " + str(eta) + " and omega: " + str(omega))

                    # Find proper omega for new eta. Use old policy parameters first.
                    set_from_flat(pi.theta_beta_to_all(prev_theta, prev_beta))
                    eta, omega = \
                        eta_omega_optimizer.optimize(w_theta, Waa, Wsa, wa, varphis, pi.get_kt(),
                                                     pi.get_prec_matrix(), pi.is_new_policy_valid, old_ent, eta)
                    logger.log("Updated omega, eta: " + str(eta) + " and omega: " + str(omega))

                # Use final policy
                logger.log("Final eta: " + str(eta) + " and omega: " + str(omega))
                cur_theta = (eta * prev_theta + w_theta.reshape(-1, )) / (eta + omega)
                cur_beta = prev_beta + w_beta.reshape(-1, ) / eta
                set_from_flat(pi.theta_beta_to_all(cur_theta, cur_beta))

                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))

            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("Name", method)
        logger.record_tabular("Iteration", iters_so_far)
        logger.record_tabular("trial", trial)

        if rank==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]