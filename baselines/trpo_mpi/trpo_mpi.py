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

def learn(env, policy_fn, *,
        timesteps_per_batch, # what to train on
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        i_trial):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    # policy and guided policy
    pi = policy_fn("pi", ob_space, ac_space)
    # define guided policy
    gpi = policy_fn("gpi", ob_space, ac_space)

    # old policy and old guided policy
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    goldpi = policy_fn("goldpi", ob_space, ac_space)

    # Target advantage function and return
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    gatarg = tf.placeholder(dtype=tf.float32, shape=[None])
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    gret = tf.placeholder(dtype=tf.float32, shape=[None])

    ob = U.get_placeholder_cached(name="ob")
    gob = U.get_placeholder_cached(name='gob')

    # Sample action according to policy
    ac = pi.pdtype.sample_placeholder([None])
    gac = gpi.pdtype.sample_placeholder([None])

    # Kl divergence
    kloldnew = oldpi.pd.kl(pi.pd)
    # change the oder here
    gkloldnew = goldpi.pd.kl(gpi.pd)

    # Entropy of policy and guided policy
    ent = pi.pd.entropy()
    gent = gpi.pd.entropy()

    # Mean kl divergence of origin policy and guided policy
    meankl = tf.reduce_mean(kloldnew)
    gmeankl = tf.reduce_mean(gkloldnew)

    # Mean entropy and entropy penalty term
    meanent = tf.reduce_mean(ent)
    gmeanent = tf.reduce_mean(gent)
    entbonus = entcoeff * meanent
    gentbonus = entcoeff * gmeanent

    # Value function error consists of
    # (value of gpi - target value of pi)^2 and
    # (value of pi - target value of gpi)^2
    vferr = tf.reduce_mean(tf.square(pi.vpred - gret))
    gvferr = tf.reduce_mean(tf.square(gpi.vpred - ret))

    # define different ratio here
    ratio = tf.exp(pi.pd.logp(gac) - goldpi.pd.logp(gac))  # with gaction
    gratio = tf.exp(pi.pd.logp(ac) - goldpi.pd.logp(ac))  # advantage * pnew / gpold with action

    surrgain = tf.reduce_mean(ratio * gatarg) # with advantage target
    gsurrgain = tf.reduce_mean(gratio * atarg) # with guided advantage target (guided action)

    optimgain = surrgain + entbonus
    goptimgain = gsurrgain + gentbonus

    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    glosses = [goptimgain, gmeankl, gentbonus, gsurrgain, gmeankl]

    loss_names = ["optimgain", "meankl","entloss", "surrgain", "entropy"]
    glosses_names = ["goptimgain", "gmeankl", "gentloss", "gsurrgain", "gentropy"]

    dist = meankl
    gdist = gmeankl

    all_var_list = pi.get_trainable_variables()
    gall_var_list = gpi.get_trainable_variables()

    # policy variable list
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    gvar_list = [v for v in gall_var_list if v.name.split("/")[1].startswith("pol")]

    # value function list
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    gvf_var_list = [v for v in gall_var_list if v.name.split("/")[1].startswith("vf")]

    vfadam = MpiAdam(vf_var_list)
    gvfadam = MpiAdam(gvf_var_list)

    get_flat = U.GetFlat(var_list)
    gget_flat = U.GetFlat(gvar_list)

    set_from_flat = U.SetFromFlat(var_list)
    gset_from_flat = U.SetFromFlat(gvar_list)

    klgrads = tf.gradients(dist, var_list)
    gklgrads = tf.gradients(gdist, gvar_list)


    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    gflat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="gflat_tan")

    shapes = [var.get_shape().as_list() for var in var_list]
    gshapes = [var.get_shape().as_list() for var in gvar_list]

    start = 0
    gstart = 0

    tangents = []
    gtangents = []

    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz

    for shape in gshapes:
        sz = U.intprod(shape)
        gtangents.append(tf.reshape(gflat_tangent[gstart:gstart+sz], shape))
        gstart += sz

    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    ggvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(gklgrads, gtangents)])

    fvp = U.flatgrad(gvp, var_list)
    gfvp = U.flatgrad(ggvp, gvar_list)


    # assign old parameters to new parameters
    def assign(old, new):
        return U.function([], [], updates=[tf.assign(oldv, newv)
             for (oldv, newv) in zipsame(old.get_variables(), new.get_variables())])


    assign_old_eq_new = assign(oldpi, pi)
    gassign_old_eq_new = assign(goldpi, gpi)

    compute_losses = U.function([ob, ac, atarg], losses)
    compute_glosses = U.function([gob, gac, gatarg], glosses)

    compute_lossandgrad = U.function([gob, gac, gatarg], glosses + [U.flatgrad(goptimgain, var_list)])
    compute_glossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, gvar_list)])

    # Fisher vector product of policy and guided policy
    compute_fvp = U.function([flat_tangent, gob, gac, gatarg], fvp)
    compute_gfvp = U.function([gflat_tangent, ob, ac, atarg], gfvp)

    # Value function loss & gradients for policy and guided policy
    compute_vflossandgrad = U.function([gob, gret], U.flatgrad(gvferr, vf_var_list))
    compute_gvflossandgrad = U.function([ob, ret], U.flatgrad(vferr, gvf_var_list))

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

    # Initialize all parameters
    U.initialize()
    gth_init = gget_flat()
    MPI.COMM_WORLD.Bcast(gth_init, root=0)
    set_from_flat(gth_init)
    gvfadam.sync()
    print("Init param sum of guided policy and value net", gth_init.sum(), flush=True)


    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum of training policy and value net", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)
    gseg_gen = traj_segment_generator(gpi, env, timesteps_per_batch, stochastic=True)

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
        # logger.log("********** Iteration %i ************"%iters_so_far)
        print("********** Iteration %i ************"%iters_so_far)

        ############################################################
        ################## Guided Policy Training Part #############
        ############################################################

        # generate samples with policy net used for training guided policy net
        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]
        atarg = (atarg - atarg.mean()) / atarg.std()

        if hasattr(pi, "ret_rms"): gpi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): gpi.ob_rms.update(ob) # update running mean/std for policy

        # set arguments of guided policy
        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]

        # define fisher vector product function (input is g)
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        # set old parameter values to new parameter values
        gassign_old_eq_new()
        with timed("computegrad"):
            *glossbefore, gg = compute_glossandgrad(*args)
        glossbefore = allmean(np.array(glossbefore))
        gg = allmean(gg)

        if np.allclose(gg, 0):
            # logger.log("Got zero gradient. not updating")
            print("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
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
                # logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                print("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    # logger.log("Got non-finite value of losses -- bad!")
                    print("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    # logger.log("violated KL constraint. shrinking step.")
                    print("violated KL constraint. shrinking step.")
                elif improve < 0:
                    # logger.log("surrogate didn't improve. shrinking step.")
                    print("surrogate didn't improve. shrinking step.")
                else:
                    # logger.log("Stepsize OK!")
                    print("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                # logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.logkv(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                         include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.logkv("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.logkv("EpLenMean", np.mean(lenbuffer))
        logger.logkv("EpRewMean", np.mean(rewbuffer))
        logger.logkv("EpThisIter", len(lens))
        logger.logkv('trial', i_trial)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.logkv("EpisodesSoFar", episodes_so_far)
        logger.logkv("TimestepsSoFar", timesteps_so_far)
        logger.logkv("TimeElapsed", time.time() - tstart)
        logger.logkv("Iteration", iters_so_far)

        if rank == 0:
            logger.dumpkvs()


        ############################################################
        ################## Policy Training Part ####################
        ############################################################

        # generate guided samples, compute advantage, td value to train policy&value net
        with timed("gsampling"):
            gseg = gseg_gen.__next__()
        add_vtarg_and_adv(gseg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        gob, gac, gatarg, gtdlamret = gseg["ob"], gseg["ac"], gseg["adv"], gseg["tdlamret"]
        gvpredbefore = gseg["vpred"] # predicted value function before udpate
        gatarg = (gatarg - gatarg.mean()) / gatarg.std() # standardized advantage function estimate

        # use guided smaples to update value net and policy net
        if hasattr(pi, "ret_rms"): pi.ret_rms.update(gtdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(gob) # update running mean/std for policy

        # calculate arguments of policy
        gargs = gseg["ob"], gseg["ac"], gatarg
        gfvpargs = [arr[::5] for arr in gargs]

        # define fisher vector product function (input is g)
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        # set old parameter values to new parameter values
        assign_old_eq_new()
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            # logger.log("Got zero gradient. not updating")
            print("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
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
                # logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                print("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    # logger.log("Got non-finite value of losses -- bad!")
                    print("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    # logger.log("violated KL constraint. shrinking step.")
                    print("violated KL constraint. shrinking step.")
                elif improve < 0:
                    # logger.log("surrogate didn't improve. shrinking step.")
                    print("surrogate didn't improve. shrinking step.")
                else:
                    # logger.log("Stepsize OK!")
                    print("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                # logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.logkv(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                         include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.logkv("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.logkv("EpLenMean", np.mean(lenbuffer))
        logger.logkv("EpRewMean", np.mean(rewbuffer))
        logger.logkv("EpThisIter", len(lens))
        logger.logkv('trial', i_trial)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.logkv("EpisodesSoFar", episodes_so_far)
        logger.logkv("TimestepsSoFar", timesteps_so_far)
        logger.logkv("TimeElapsed", time.time() - tstart)
        logger.logkv("Iteration", iters_so_far)

        if rank == 0:
            logger.dumpkvs()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]