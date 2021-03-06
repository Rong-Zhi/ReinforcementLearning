#!/usr/bin/env python3
import argparse
# from baselines.common.cmd_util import mujoco_arg_parser
import sys
# sys.path.append('/work/scratch/rz97hoku/ReinforcementLearning')
# sys.path.append('/home/zhi/Documents/ReinforcementLearning/')
sys.path.append('/Users/zhirong/Documents/ReinforcementLearning/')
from baselines.common.cmd_util import control_arg_parser, make_control_env
from baselines import bench, logger
import os
import os.path as osp
import timeit
import datetime

from baselines.common import set_global_seeds

from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import gym
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from baselines.env.box2d.lunar_lander_pomdp import LunarLanderContinuousPOMDP
from baselines.env.envsetting import newenv
import mpi4py as MPI


def train(env_id, num_timesteps, seed, nsteps, batch_size, epoch,
          method, net_size, i_trial, load_path, use_entr, ncpu):
    # rank = MPI.COMM_WORLD.Get_rank()
    # if rank != 0:
    #     logger.set_level(logger.DISABLED)

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True

    # workerseed = seed + 10000 * rank
    tf.reset_default_graph()
    set_global_seeds(seed)


    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            if logger.get_dir():
                env = bench.Monitor(env, os.path.join(logger.get_dir(), 'train-{}.monitor.json'.format(rank)))
            return env
        return _thunk

    # def make_env():
    #     env = gym.make(env_id)
    #     env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    #     return env

    env = SubprocVecEnv([make_env(i) for i in range(ncpu)])
    # env = DummyVecEnv([make_env])
    env = VecNormalize(env)
    with tf.Session(config=config) as sess:
        policy = MlpPolicy
        ppo2.learn(policy=policy, env=env, nsteps=nsteps, nminibatches=batch_size,
            lam=0.95, gamma=0.99, noptepochs=epoch, log_interval=1,
            ent_coef=0.01, lr=3e-4, cliprange=0.2,
            total_timesteps=num_timesteps, useentr=use_entr, net_size=net_size,
            i_trial=i_trial, load_path=load_path, method=method)

def render(env_id, nsteps, batch_size, net_size, load_path, video_path, iters):
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env
    env = SubprocVecEnv([make_env])
    env = VecNormalize(env)
    with tf.Session() as sess:
        policy = MlpPolicy
        ppo2.render(policy=policy, env=env, nsteps=nsteps, lam=0.95, gamma=0.99, nminibatches=batch_size,
                    net_size=net_size, load_path=load_path, video_path=video_path, iters_so_far=iters)

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def save_args(args):
    for arg in vars(args):
        logger.logkv("{}".format(arg), getattr(args, arg))
    logger.dumpkvs()

def main():
    args = control_arg_parser().parse_args()
    if args.env == 'LunarLanderContinuousPOMDP-v0':
        newenv(hist_len=args.hist_len)
    if args.train is True:
        ENV_path = get_dir(os.path.join(args.log_dir, args.env))
        log_dir = os.path.join(ENV_path, args.method +"-"+
                               '{0}'.format(args.seed))+"-" +\
                  datetime.datetime.now().strftime("%m-%d-%H-%M")

        # if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure(dir=log_dir)
        save_args(args)
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
              nsteps=args.nsteps, batch_size=args.batch_size, epoch=args.epoch,
              method=args.method, net_size=args.net_size, ncpu=args.ncpu,
              i_trial=args.seed, load_path=args.load_path, use_entr=int(args.use_entr))
    if args.render is True:
        video_path = osp.split(osp.split(args.load_path)[0])[0]
        render(args.env, nsteps=args.nsteps, batch_size=args.batch_size, net_size=args.net_size,
               load_path=args.load_path, video_path=video_path, iters=args.iters)

if __name__ == '__main__':
    main()
