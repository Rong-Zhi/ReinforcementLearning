import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
import imageio


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, vf_coef, max_grad_norm, net_size):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, net_size, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, net_size, reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        ENT_DYNAMIC = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())


        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ENT_DYNAMIC + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, ent_dynamic,states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values, ENT_DYNAMIC:ent_dynamic}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'loss_ent', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def play(self, path, iters_so_far):
        obs = self.env.reset()
        state = self.model.initial_state
        num_episodes = 0
        done = False
        frames = []
        rwds = []
        while True:
            frame = self.env.unwrapedrender()
            frames.append(frame)
            action, _, states, _ = self.model.step(obs, state, done)
            obs, reward, done, _ = self.env.step(action)
            rwds.append(reward)
            if done:
                print("Saved video.")
                imageio.mimsave(path + '/' + str(iters_so_far) + '.gif', frames, fps=20)
                break
            num_episodes += 1
        return rwds


    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)




def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=200, useentr, net_size, load_path=None, i_trial, method):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, net_size=net_size)
    if save_interval:
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))

    model = make_model()

    if load_path:
        model.load(load_path=load_path)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch


    # ent_coef = max(ent_coef - 0.25*float(update) / float(nupdates), 0.001)
    ent_coef = useentr * ent_coef
    # ent_coef = entp - float(iters_so_far) / float(max_iters)

    for update in range(1, nupdates+1):
        # ent_coef = useentr * 0.01
        # ent_coef = max(ent_coef - 0.25 * float(update) / float(nupdates), 0.001)
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, ent_dynamic=ent_coef))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates, ent_dynamic=ent_coef))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('EpRewMean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('EpLenMean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            logger.logkv('trial', i_trial)
            logger.logkv("Iteration", update)
            logger.logkv('Name', method)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()

        # if update == 1 or update % 100==0 or update==nupdates:
        #     rwd=runner.play(video_path=logger.get_dir()+'/videos', iters_so_far=update)
        #     print('Average Retrun:{0}'.format(np.sum(rwd)/float(len(rwd))))
        #     print('Sum of Return:{0}'.format(np.sum(rwd)))

        if save_interval and (update % save_interval == 0 or update == 1 or update==nupdates) and logger.get_dir():
            checkdir = get_dir(osp.join(logger.get_dir(), 'checkpoints'))
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
            # np.save('{}/mean'.format(checkdir + '/'), runner.env.obs.mean)
            # np.save('{}/var'.format(checkdir + '/'), runner.env.obs.var)
    env.close()

def render(*, policy, env, nsteps, vf_coef=0.5,  max_grad_norm=0.5,
           gamma=0.99, lam=0.95, nminibatches=4, net_size, load_path=None, iters_so_far, video_path):

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                               nbatch_train=nbatch_train,
                               nsteps=nsteps, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm, net_size=net_size)
    model = make_model()
    model.load(load_path=load_path)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    path = get_dir(osp.join(video_path, 'videos'))
    rwd = runner.play(path=path, iters_so_far=iters_so_far)
    print("Average Return:{0}".format(np.sum(rwd)))
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
