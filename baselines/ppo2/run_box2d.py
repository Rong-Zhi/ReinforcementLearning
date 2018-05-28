#!/usr/bin/env python3
import argparse
# from baselines.common.cmd_util import mujoco_arg_parser
import sys
sys.path.append('/work/scratch/rz97hoku/ReinforcementLearning/')
# sys.path.append('/home/zhi/Documents/ReinforcementLearning/')
# sys.path.append('/Users/zhirong/Documents/ReinforcementLearning/')
from baselines.common.cmd_util import control_arg_parser, make_control_env
from baselines import bench, logger
import os
import os.path as osp
import timeit
import datetime

from baselines.common import set_global_seeds

from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy, mdPolicy
import gym
import tensorflow as tf
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import  DummyVecEnv
from baselines.env.envsetting import newenv


def train(env_id, num_timesteps, seed, nsteps, batch_size, epoch, hist_len, env_name, policy_name,
          method, net_size, load_path, use_entr, ncpu, rank, checkpoint, filter_size):
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)

    config.gpu_options.allow_growth = True
    tf.reset_default_graph()

    workerseed = seed*1000

    # one should always use same definition for train and render

    # def make_env():
    #     env = gym.make(env_id)
    #     env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    #     return env
    # env = DummyVecEnv([make_env])
    # env = VecNormalize(env)

    def make_env(icpu):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + icpu)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), 'train-{}-monitor'.format(icpu)))
            return env
        return _thunk

    env = SubprocVecEnv([make_env(i) for i in range(ncpu)])
    env = VecNormalize(env)

    set_global_seeds(workerseed)
    with tf.Session(config=config) as sess:
        if policy_name =='mdPolicy':
            policy = mdPolicy
        else:
            policy = MlpPolicy
        ppo2.learn(policy=policy, env=env, nsteps=nsteps, nminibatches=batch_size,
            lam=0.95, gamma=0.99, noptepochs=epoch, log_interval=1, env_name=env_name,
            ent_coef=0.1, lr=3e-4, cliprange=0.2, hist_len=hist_len, policy_name=policy_name,
            total_timesteps=num_timesteps, useentr=use_entr, net_size=net_size, filter_size=filter_size,
            i_trial=rank, load_path=load_path, method=method, checkpoint=checkpoint)

def render(env_id, nsteps, batch_size, net_size, load_path,
           video_path, iters, hist_len, policy_name, env_name, filter_size):


    # SubprocVecEnv, render function hasn't been setup yet
    # def make_env(seed):
    #     def _thunk():
    #         env = gym.make(env_id)
    #         env.seed(seed)
    #         env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), 'render-result'), allow_early_resets=True)
    #         return env
    #     return _thunk
    # env = SubprocVecEnv([make_env(seed=icpu) for icpu in range(ncpu)])

    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, os.path.join(video_path, 'render-result'), allow_early_resets=True)
        return env
    env = DummyVecEnv([make_env])

    env = VecNormalize(env)
    with tf.Session() as sess:
        policy = MlpPolicy
        ppo2.render(policy=policy, env=env, nsteps=nsteps, lam=0.95, gamma=0.99, nminibatches=batch_size,
                    policy_name=policy_name, env_name=env_name, net_size=net_size,
                    load_path=load_path, video_path=video_path, iters_so_far=iters, hist_len=hist_len)

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def save_args(args):
    for arg in vars(args):
        logger.log("{}:".format(arg), getattr(args, arg))
    # logger.dumpkvs()

def main():
    args = control_arg_parser().parse_args()
    # rank = MPI.COMM_WORLD.Get_rank()
    # print("This is rank:", rank, "Seed:", args.seed)
    if args.env == 'LunarLanderContinuousPOMDP-v0':
        newenv(hist_len=args.hist_len, block_high=float(args.block_high), policy_name=args.policy_name)
    if args.train is True:
        ENV_path = get_dir(os.path.join(args.log_dir, args.env))
        log_dir = os.path.join(ENV_path, args.method +"-"+
                               '{}'.format(args.seed))+"-" +\
                  datetime.datetime.now().strftime("%m-%d-%H-%M")
        logger.configure(dir=log_dir)
        # logger.log("This is rank {}".format(rank))
        save_args(args)
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, hist_len=args.hist_len,
              nsteps=args.nsteps, batch_size=args.batch_size, epoch=args.epoch, env_name=args.env,
              method=args.method, net_size=tuple(args.net_size), ncpu=args.ncpu, policy_name=args.policy_name,
              load_path=args.load_path, use_entr=int(args.use_entr), rank=args.seed,
              checkpoint=args.checkpoint, filter_size=args.filter_size)

    if args.render is True:
        video_path = osp.split(osp.split(args.load_path)[0])[0]
        logger.configure(dir=video_path)
        render(args.env, nsteps=args.nsteps, batch_size=args.batch_size, net_size=args.net_size,
               policy_name=args.policy_name, env_name=args.env, load_path=args.load_path,
               video_path=video_path, iters=args.iters, hist_len=args.hist_len, filter_size=args.filter_size)

if __name__ == '__main__':
    main()
