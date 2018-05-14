#!/usr/bin/env python3
import argparse
# from baselines.common.cmd_util import mujoco_arg_parser
import sys
sys.path.append('/work/scratch/rz97hoku/ReinforcementLearning')
# sys.path.append('/home/zhi/Documents/ReinforcementLearning/')
# sys.path.append('/Users/zhirong/Documents/Masterthesis-code/')
from baselines.common.cmd_util import control_arg_parser, make_control_env
from baselines import bench, logger
import os
import os.path as osp
import timeit
import datetime

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import gym
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.env.lunar_lander_pomdp import LunarLanderContinuousPOMDP


def train(env_id, num_timesteps, seed, nsteps, batch_size, epoch,
          method, hist_len, net_size, i_trial):
    ncpu = 4
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True


    def make_env():
        if env_id == 'LunarLanderContinuousPOMDP-v0':
            env = LunarLanderContinuousPOMDP(hist_len=hist_len)
        else:
            env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

        tf.reset_default_graph()
        set_global_seeds(seed)
        env = DummyVecEnv([make_env])
        env = VecNormalize(env)

        # with tf.Session() as sess:
        with tf.Session(config=config) as sess:
            policy = MlpPolicy
            ppo2.learn(policy=policy, env=env, nsteps=nsteps, nminibatches=batch_size,
                lam=0.95, gamma=0.99, noptepochs=epoch, log_interval=1,
                ent_coef=0.01, lr=3e-4, cliprange=0.2,
                total_timesteps=num_timesteps, useentr=True, net_size=net_size,
                i_trial=i_trial)

def render(env_id, nsteps, batch_size, hist_len, net_size, load_path):

    def make_env():
        if env_id == 'LunarLanderContinuousPOMDP-v0':
            env = LunarLanderContinuousPOMDP(hist_len=hist_len)
        else:
            env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)
    with tf.Session() as sess:
        policy = MlpPolicy
        ppo2.render(policy=policy, env=env, nsteps=nsteps, lam=0.95, gamma=0.99, nminibatches=batch_size,
                    net_size=net_size, load_path=load_path, iters_so_far=0)

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def main():
    args = control_arg_parser().parse_args()
    if args.train:
        ENV_path = get_dir(os.path.join(args.log_dir, args.env))
        log_dir = os.path.join(ENV_path, args.method +"-"+
                               '{0}'.format(args.seed))+"-" +\
                  datetime.datetime.now().strftime("%m-%d-%H-%M")

        logger.configure(dir=log_dir)
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
              nsteps=args.nsteps, batch_size=args.batch_size, epoch=args.epoch,
              method=args.method, hist_len=args.hist_len,net_size=args.net_size,
              i_trial=args.seed)
    if args.render:
        log_dir = osp.split(osp.split(args.load_path)[0])[0]
        logger.configure(dir=log_dir)
        render(args.env, nsteps=args.nsteps, batch_size=args.batch_size, hist_len=args.hist_len,
               net_size=args.net_size, load_path=args.load_path)

if __name__ == '__main__':
    main()
