#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
import sys
import imageio

# sys.path.append('/work/scratch/rz97hoku/ReinforcementLearning/')
# sys.path.append('/home/zhi/Documents/ReinforcementLearning/')
sys.path.append('/Users/zhirong/Documents/ReinforcementLearning/')

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import arg_parser
from baselines.guided.compatible_mlp_policy import CompatibleMlpPolicy
# from baselines.guided.compatible_cnn_policy import CompatiblecnnPolicy
from baselines.guided import copos_mpi
from baselines.env.envsetting import newenv

import tensorflow as tf
import numpy as np

import gym
from baselines.common.cmd_util import control_arg_parser, make_control_env
# from baselines import bench, logger
import os
import os.path as osp
# import timeit
import datetime



def train_copos(env_id, num_timesteps, seed, trial, hist_len, block_high,
                nsteps, method, hid_size, give_state, vf_iters):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    workerseed = seed * 10000
    def policy_fn(name, ob_space, ac_space, ob_name):
        return CompatibleMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, num_hid_layers=2, ob_name=ob_name)

    set_global_seeds(workerseed)
    env = make_control_env(env_id, seed, hist_len=hist_len,
                           block_high=block_high, version0=False, give_state=give_state)
    env.seed(workerseed)

    timesteps_per_batch=nsteps
    beta = -1
    if beta < 0:
        nr_episodes = num_timesteps // timesteps_per_batch
        # Automatically compute beta based on initial entropy and number of iterations
        tmp_pi = policy_fn("tmp_pi", env.observation_space, env.action_space, ob_name="tmp_ob")
        sess.run(tf.global_variables_initializer())

        tmp_ob = np.zeros((1,) + env.observation_space.shape)
        entropy = sess.run(tmp_pi.pd.entropy(), feed_dict={tmp_pi.ob: tmp_ob})
        beta = 2 * entropy / nr_episodes
        print("Initial entropy: " + str(entropy) + ", episodes: " + str(nr_episodes))
        print("Automatically set beta: " + str(beta))

    copos_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, epsilon=0.01, beta=beta,
                    cg_iters=10, cg_damping=0.1, method=method,
                    max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=vf_iters, vf_stepsize=1e-3,
                    trial=trial, crosskl_coeff=0.01, kl_target=0.01, sess=sess)
    env.close()

def render(hid_size, load_path, video_path, env_id, seed, hist_len, block_high, give_state):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()
    def policy_fn(name, ob_space, ac_space, ob_name):
        return CompatibleMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, num_hid_layers=2, ob_name=ob_name)

    env = make_control_env(env_id, seed, hist_len=hist_len,
                           block_high=block_high, version0=True, give_state=give_state)
    pi = policy_fn("pi", env.observation_space, env.action_space, ob_name="ob")
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, load_path)
    ob = env.reset()
    frames = []
    while True:
        frame = env.unwrapped.render(mode='rgb_array')
        frames.append(frame)
        ac, vpred = pi.act(stochastic=False, ob=ob)
        print(ob)
        ob, rwd, done, _ = env.step(ac)
        if done:
            imageio.mimsave(video_path+'result.mp4', frames, fps=20)
            break
    env.close()

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def save_args(args):
    for arg in vars(args):
        logger.log("{}:".format(arg), getattr(args, arg))

def main():
    args = control_arg_parser().parse_args()

    # Training

    ENV_path = get_dir(os.path.join(args.log_dir, args.env))
    log_dir = os.path.join(ENV_path, args.method + "-" +
                           '{}'.format(args.seed)) + "-" + \
              datetime.datetime.now().strftime("%m-%d-%H-%M")
    logger.configure(dir=log_dir)
    save_args(args)
    train_copos(args.env, num_timesteps=args.num_timesteps * 1e6, seed=args.seed, trial=args.seed,
                hist_len=args.hist_len, block_high=float(args.block_high), nsteps=args.nsteps,
                method=args.method, hid_size=args.hid_size, give_state=bool(args.give_state), vf_iters=args.epoch)

    #Render

    # load_path = '/Users/zhirong/Documents/ReinforcementLearning/tmp/LunarLanderContinuousPOMDP-v0/copos-guided-try-diffinput-0-07-05-13-34/checkpoints/00976.ckpt'
    # video_path = '/Users/zhirong/Documents/ReinforcementLearning/tmp/LunarLanderContinuousPOMDP-v0/copos-guided-try-diffinput-0-07-05-13-34/'
    # render(hid_size=args.hid_size,load_path=load_path, video_path=video_path,
    #        env_id=args.env, seed=0, hist_len=args.hist_len, block_high=args.block_high,
    #        give_state=0)


if __name__ == '__main__':
    main()

