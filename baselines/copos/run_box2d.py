#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import arg_parser
from baselines.copos.compatible_mlp_policy import CompatibleMlpPolicy
from baselines.copos import copos_mpi
from baselines.env.envsetting import newenv

import tensorflow as tf
import numpy as np

import gym
import sys

sys.path.append('/work/scratch/rz97hoku/ReinforcementLearning/')
# sys.path.append('/home/zhi/Documents/ReinforcementLearning/')
# sys.path.append('/Users/zhirong/Documents/ReinforcementLearning/')
from baselines.common.cmd_util import control_arg_parser, make_control_env
# from baselines import bench, logger
import os
import os.path as osp
# import timeit
import datetime

def train_copos(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    workerseed = seed * 10000
    def policy_fn(name, ob_space, ac_space):
        return CompatibleMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)

    set_global_seeds(workerseed)
    # env = gym.make(env_id)

    env = make_control_env(env_id, seed, hist_len=None)
    env.seed(workerseed)

    timesteps_per_batch=2048
    beta = -1
    if beta < 0:
        nr_episodes = num_timesteps // timesteps_per_batch
        # Automatically compute beta based on initial entropy and number of iterations
        tmp_pi = policy_fn("tmp_pi", env.observation_space, env.action_space)

        sess.run(tf.global_variables_initializer())

        tmp_ob = np.zeros((1,) + env.observation_space.shape)
        entropy = sess.run(tmp_pi.pd.entropy(), feed_dict={tmp_pi.ob: tmp_ob})
        beta = 2 * entropy / nr_episodes
        print("Initial entropy: " + str(entropy) + ", episodes: " + str(nr_episodes))
        print("Automatically set beta: " + str(beta))

    copos_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, epsilon=0.01, beta=beta, cg_iters=10, cg_damping=0.1,
                    max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def argparser():
    """
    Create an argparse.ArgumentParser.
    """
    parser = arg_parser()
    parser.add_argument('--log_dir', type=str, default='/home/zhi/Documents/ReinforcementLearning/tmp')
    parser.add_argument('--env', help='environment ID', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--method', help='method', type=str, default='copos-try')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(2e6))
    return parser


def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def save_args(args):
    for arg in vars(args):
        logger.log("{}:".format(arg), getattr(args, arg))

def main():
    args = argparser().parse_args()
    ENV_path = get_dir(os.path.join(args.log_dir, args.env))
    log_dir = os.path.join(ENV_path, args.method + "-" +
                           '{}'.format(args.seed)) + "-" + \
              datetime.datetime.now().strftime("%m-%d-%H-%M")
    logger.configure(dir=log_dir)
    save_args(args)
    # if args.env == 'LunarLanderContinuousPOMDP-v0':
    #     newenv(hist_len=args.hist_len, block_high=float(args.block_high), policy_name=args.policy_name)
    train_copos(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()

