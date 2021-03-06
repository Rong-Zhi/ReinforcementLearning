from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

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

def learn(env, genv, i_trial,policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    gpi = policy_fn("gpi", ob_space, ac_space) # Construct network for new policy
    goldpi = policy_fn("goldpi", ob_space, ac_space) # Network for old policy
    gatarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    gret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    # gob = U.get_placeholder_cached(name='ob')
    gac = gpi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    gkloldnew = goldpi.pd.kl(gpi.pd)
    gent = gpi.pd.entropy()
    gmeankl = tf.reduce_mean(gkloldnew)
    gmeanent = tf.reduce_mean(gent)
    gpol_entpen = (-entcoeff) * gmeanent


    ratio = tf.exp(pi.pd.logp(gac) - goldpi.pd.logp(gac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    gratio = tf.exp(gpi.pd.logp(ac) - oldpi.pd.logp(ac))
    gsurr1 = gratio * gatarg
    gsurr2 = tf.clip_by_value(gratio, 1.0 - clip_param, 1.0 + clip_param) * gatarg
    gpol_surr = - tf.reduce_mean(tf.minimum(gsurr1, gsurr2))
    gvf_loss = tf.reduce_mean(tf.square(gpi.vpred - gret))
    gtotal_loss = gpol_surr + gpol_entpen + gvf_loss
    glosses = [gpol_surr, gpol_entpen, gvf_loss, gmeankl, gmeanent]
    gloss_names = ["gpol_surr", "gpol_entpen", "gvf_loss", "gkl", "gent"]


    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, gac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    gvar_list = gpi.get_trainable_variables()
    glossandgrad = U.function([ob, ac, gatarg, gret, lrmult], glosses + [U.flatgrad(gtotal_loss, gvar_list)])
    gadam = MpiAdam(gvar_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])

    gassign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(goldpi.get_variables(), gpi.get_variables())])


    compute_losses = U.function([ob, gac, atarg, ret, lrmult], losses)
    gcompute_losses = U.function([ob, ac, gatarg, gret, lrmult], glosses)


    U.initialize()
    adam.sync()
    gadam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, gpi, env, timesteps_per_actorbatch, stochastic=True)
    gseg_gen = traj_segment_generator(gpi, pi, genv, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    glenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    grewbuffer = deque(maxlen=100)


    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    def standarize(value):
        return (value - value.mean()) / (value.std())

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        print("********** Iteration %i ************"%iters_so_far)

        print("********** Guided Policy ************")

        gseg = gseg_gen.__next__()
        add_vtarg_and_adv(gseg, gamma, lam)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)



        gob, gac, gatarg, gatarg_, gtdlamret, gtdlamret_ , gvpredbefore, gvpredbefore_ = gseg["ob"], gseg["ac"], \
                                gseg["adv"], gseg["adv_"], gseg["tdlamret"], gseg["tdlamret_"], gseg["vpred"], gseg["vpred_"]

        standarize(gatarg_)
        standarize(gatarg)

        gd = Dataset(dict(gob=gob, gac=gac, gatarg=gatarg, gatarg_=gatarg_, gvtarg=gtdlamret, gvtarg_=gtdlamret_),
                     shuffle=not gpi.recurrent)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, atarg_, tdlamret, tdlamret_, vpredbefore, vpredbefore_ = seg["ob"], seg["ac"],\
                            seg["adv"], seg["adv_"], seg["tdlamret"], seg["tdlamret_"], seg["vpred"], gseg["vpred_"]

        standarize(atarg)
        standarize(atarg_)

        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, atarg_=atarg_, vtarg=tdlamret, vtarg_=tdlamret_),
                    shuffle=not pi.recurrent)

        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(gpi, "ob_rms"): gpi.ob_rms.update(ob)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(gob) # update running mean/std for policy

        gassign_old_eq_new()
        print("Optimizing...Guided Policy")
        # print(fmt_row(13, gloss_names))

        # Here we do a bunch of optimization epochs over the data

        for _ in range(optim_epochs):
            glosses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = glossandgrad(batch["ob"], batch["ac"], batch["atarg_"], batch["vtarg_"], cur_lrmult)
                gadam.update(g, optim_stepsize * cur_lrmult)
                glosses.append(newlosses)
            # print(fmt_row(13, np.mean(glosses, axis=0)))

        # print("Evaluating losses...")
        glosses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = gcompute_losses(batch["ob"], batch["ac"], batch["atarg_"], batch["vtarg_"], cur_lrmult)
            glosses.append(newlosses)
        gmeanlosses, _, _ = mpi_moments(glosses, axis=0)
        # print(fmt_row(13, gmeanlosses))

        for (lossval, name) in zipsame(gmeanlosses, gloss_names):
            logger.record_tabular("gloss_" + name, lossval)
        # logger.record_tabular("gev_tdlam_before", explained_variance(vpredbefore, tdlamret))


        assign_old_eq_new() # set old parameter values to new parameter values
        print("Optimizing...Training Policy")
        # print(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data

        optim_batchsize = optim_batchsize or gob.shape[0]


        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in gd.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["gob"], batch["gac"], batch["gatarg_"], batch["gvtarg_"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            # print(fmt_row(13, np.mean(losses, axis=0)))

        # print("Evaluating losses...")
        losses = []
        for batch in gd.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["gob"], batch["gac"], batch["gatarg_"], batch["gvtarg_"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        # print(fmt_row(13, meanlosses))

        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))


        glrlocal = (gseg["ep_lens"], gseg["ep_rets"]) # local values
        glistoflrpairs = MPI.COMM_WORLD.allgather(glrlocal) # list of tuples
        glens, grews = map(flatten_lists, zip(*glistoflrpairs))


        # lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        grewbuffer.extend(grews)
        # logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("GEpRewMean", np.mean(grewbuffer))
        # logger.record_tabular("EpThisIter", len(lens))



        # episodes_so_far += len(lens)
        # timesteps_so_far += sum(lens)
        iters_so_far += 1
        # logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        logger.logkv('trial', i_trial)
        logger.logkv("Iteration", iters_so_far)

        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
