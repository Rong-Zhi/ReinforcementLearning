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


def traj_segment_generator(pi, pi_, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    vpreds_ = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        ac_, vpred_ = pi_.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "vpred_": vpreds_, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new), "nextvpred_": vpred_ * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        vpreds_[i] = vpred_
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
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    vpred_ = np.append(seg["vpred_"], seg["nextvpred_"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    seg["adv_"] = gaelam_ = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    l = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        delta_ = rew[t] + gamma * vpred_[t+1] * nonterminal - vpred_[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        gaelam_[t] = l = delta_ + gamma * lam * nonterminal * l

    seg["tdlamret"] = seg["adv"] + seg["vpred"]
    seg["tdlamret_"] = seg["adv_"] + seg["vpred_"]

def learn(env, genv, policy_fn, *,
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
    
    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)

    gpi = policy_fn("gpi", ob_space, ac_space)
    goldpi = policy_fn("goldpi", ob_space, ac_space)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    gatarg = tf.placeholder(dtype=tf.float32, shape=[None])
    gret = tf.placeholder(dtype=tf.float32, shape=[None])

    ob = U.get_placeholder_cached(name="ob") #check it later !!!!!!
    # gob = U.get_placeholder_cached(name="ob")

    ac = pi.pdtype.sample_placeholder([None])
    gac = gpi.pdtype.sample_placeholder([None])


    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent


    gkloldnew = goldpi.pd.kl(gpi.pd)
    gent = gpi.pd.entropy()
    gmeankl = tf.reduce_mean(gkloldnew)
    gmeanent = tf.reduce_mean(gent)
    gentbonus = entcoeff * gmeanent


    vferr = tf.reduce_mean(tf.square(pi.vpred - gret)) # check it later !!!!!!!!!!!!
    gvferr = tf.reduce_mean(tf.square(gpi.vpred - ret))


    ratio = tf.exp(pi.pd.logp(gac) - goldpi.pd.logp(gac)) # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * gatarg)
    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    gratio = tf.exp(gpi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    gsurrgain = tf.reduce_mean(gratio * atarg)
    goptimgain = gsurrgain + gentbonus
    glosses = [goptimgain, gmeankl, gentbonus, gsurrgain, gmeanent]
    gloss_names = ["goptimgain", "gmeankl", "gentloss", "gsurrgain", "gentropy"]


    dist = meankl
    gdist = gmeankl


    all_var_list = pi.get_trainable_variables()
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


    gall_var_list = gpi.get_trainable_variables()
    gvar_list = [v for v in gall_var_list if v.name.split("/")[1].startswith("pol")]
    gvf_var_list = [v for v in gall_var_list if v.name.split("/")[1].startswith("vf")]
    gvfadam = MpiAdam(gvf_var_list)

    gget_flat = U.GetFlat(gvar_list)
    gset_from_flat = U.SetFromFlat(gvar_list)
    gklgrads = tf.gradients(gdist, gvar_list)
    gflat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="gflat_tan")
    gshapes = [var.get_shape().as_list() for var in var_list]
    gstart = 0
    gtangents = []
    for shape in gshapes:
        sz = U.intprod(shape)
        gtangents.append(tf.reshape(gflat_tangent[gstart:gstart+sz], shape))
        gstart += sz
    ggvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(gklgrads, gtangents)]) #pylint: disable=E1111
    gfvp = U.flatgrad(ggvp, gvar_list)



    def assigneq(pi, oldpi):
        U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])


    compute_losses = U.function([ob, gac, gatarg], losses)
    compute_lossandgrad = U.function([ob, gac, gatarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, gac, gatarg], fvp)
    compute_vflossandgrad = U.function([ob, gret], U.flatgrad(vferr, vf_var_list))

    gcompute_losses = U.function([ob, ac, atarg], glosses)
    gcompute_lossandgrad = U.function([ob, ac, atarg], glosses + [U.flatgrad(goptimgain, gvar_list)])
    gcompute_fvp = U.function([gflat_tangent, ob, ac, atarg], gfvp)
    gcompute_vflossandgrad = U.function([ob, ret], U.flatgrad(gvferr, gvf_var_list))

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

    gth_init = gget_flat()
    MPI.COMM_WORLD.Bcast(gth_init, root=0)
    gset_from_flat(gth_init)
    gvfadam.sync()

    print("Init param sum", th_init.sum(), flush=True)
    print("Init gparam sum", gth_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, gpi, env, timesteps_per_batch, stochastic=True)
    gseg_gen = traj_segment_generator(gpi, pi, genv, timesteps_per_batch, stochastic=True)
    # gseg_gen = traj_segment_generator(gpi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    glenbuffer = deque(maxlen=40)
    grewbuffer = deque(maxlen=40)

    def standarize(value):
        return (value - value.mean()) / value.std()

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        print("********** Iteration %i ************"%iters_so_far)

        print("********** Guided Policy ************")

        with timed("gsampling"):
            gseg = gseg_gen.__next__()

        add_vtarg_and_adv(gseg, gamma, lam)
        gob, gac, gatarg, gtdlamret, gvpredbefore = gseg["ob"], gseg["ac"], gseg["adv"], gseg["tdlamret"], gseg["vpred"]
        gatarg = standarize(gatarg)



        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        ob, ac, atarg, tdlamret, vpredbefore = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["vpred"]
        atarg = standarize(atarg) # standardized advantage function estimate

        # if hasattr(gpi, "ret_rms"): gpi.ret_rms.update(tdlamret)
        if hasattr(gpi, "ob_rms"): gpi.ob_rms.update(ob) # update running mean/std for policy

        # if hasattr(pi, "ret_rms"): pi.ret_rms.update(gtdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(gob) # update running mean/std for policy

        gargs = seg["ob"], seg["ac"], atarg
        gfvpargs = [arr[::5] for arr in gargs]

        args = gseg["ob"], gseg["ac"], gatarg
        fvpargs = [arr[::5] for arr in args]

        def gfisher_vector_product(p):
            return allmean(gcompute_fvp(p, *gfvpargs)) + cg_damping * p

        assigneq(gpi, goldpi)

        with timed("gcomputegrad"):
            *glossbefore, gg = gcompute_lossandgrad(*gargs)

        glossbefore = allmean(np.array(glossbefore))
        gg = allmean(gg)

        if np.allclose(gg, 0):
            print("Got zero gradient. not updating")
        else:
            with timed("gcg"):
                gstepdir = cg(gfisher_vector_product, gg, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(gstepdir).all()

            gshs = .5*gstepdir.dot(gfisher_vector_product(gstepdir))
            glm = np.sqrt(gshs / max_kl)
            gfullstep = gstepdir / glm
            gexpectedimprove = gg.dot(gfullstep)
            gsurrbefore = glossbefore[0]
            gstepsize = 1.0
            gthbefore = gget_flat()

            # Calculate conjugate gradient and update guided theta iteratively
            for _ in range(10):
                gthnew = gthbefore + gfullstep * gstepsize
                gset_from_flat(gthnew)
                gmeanlosses = gsurr, gkl, *_ = allmean(np.array(gcompute_losses(*gargs)))
                gimprove = gsurr - gsurrbefore
                print("Expected: %.3f Actual: %.3f"%(gexpectedimprove, gimprove))
                if not np.isfinite(gmeanlosses).all():
                    print("Got non-finite value of losses -- bad!")
                elif gkl > max_kl * 1.5:
                    print("violated KL constraint. shrinking step.")
                elif gimprove < 0:
                    print("surrogate didn't improve. shrinking step.")
                else:
                    print("Stepsize OK!")
                    break
                gstepsize *= .5
            else:
                print("couldn't compute a good step")
                gset_from_flat(gthbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                gparamsums = MPI.COMM_WORLD.allgather((gthnew.sum(), gvfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, gparamsums[0]) for ps in gparamsums[1:])

        with timed("gvf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    gg = allmean(gcompute_vflossandgrad(mbob, mbret))
                    gvfadam.update(gg, vf_stepsize)

        print("********** Train Policy ************")




        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assigneq(pi, oldpi)# set old parameter values to new parameter values

        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)


        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)


        if np.allclose(g, 0):
            print("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()


            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
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
                print("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    print("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    print("violated KL constraint. shrinking step.")
                elif improve < 0:
                    print("surrogate didn't improve. shrinking step.")
                else:
                    print("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                print("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.logkv(lossname, lossval)

        for (lossname, lossval) in zip(gloss_names, gmeanlosses):
            logger.logkv(lossname, lossval)

        with timed("vf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((gseg["ob"], gseg["tdlamret"]),
                                                         include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)


        logger.logkv("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        logger.logkv("gev_tdlam_before", explained_variance(gvpredbefore, gtdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        glrlocal = (gseg["ep_lens"], seg["ep_rets"])

        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        glistoflrpairs = MPI.COMM_WORLD.allgather(glrlocal)  # list of tuples

        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        glens, grews = map(flatten_lists, zip(*glistoflrpairs))

        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        glenbuffer.extend(glens)
        grewbuffer.extend(grews)

        logger.logkv("EpLenMean", np.mean(lenbuffer))
        logger.logkv("EpRewMean", np.mean(rewbuffer))

        logger.logkv('trial', i_trial)

        logger.logkv("GEpLenMean", np.mean(glenbuffer))
        logger.logkv("GEpRewMean", np.mean(grewbuffer))

        iters_so_far += 1

        logger.logkv("TimeElapsed", time.time() - tstart)
        logger.logkv("Iteration", iters_so_far)

        if rank==0:
            logger.dumpkvs()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
