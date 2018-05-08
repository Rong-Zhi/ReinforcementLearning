#!/usr/bin/env python3
import argparse
# from baselines.common.cmd_util import mujoco_arg_parser
import sys
sys.path.append('/work/scratch/rz97hoku/ReinforcementLearning')
from baselines.common.cmd_util import control_arg_parser, make_control_env
from baselines import bench, logger
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import datetime

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import gym
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def train(env_id, num_timesteps, seed, num_trials):
    ncpu = 16
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)

    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env


    for i_trial in range(num_trials):
        tf.reset_default_graph()
        set_global_seeds(seed)
        env = DummyVecEnv([make_env])
        env = VecNormalize(env)

        with tf.Session(config=config) as sess:
            policy = MlpPolicy
            ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
                lam=0.95, gamma=0.99, noptepochs=15, log_interval=1,
                ent_coef=0.25,
                lr=3e-4,
                cliprange=0.2,
                total_timesteps=num_timesteps, useentr=True, i_trial=i_trial)

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def main():
    # args = mujoco_arg_parser().parse_args()
    args = control_arg_parser().parse_args()
    args.seed = 0
    # log_path = get_dir('/Users/zhirong/Documents/Masterthesis-code/tmp')
    # log_path = get_dir('/home/zhi/Documents/ReinforcementLearning/tmp')
    log_path = get_dir('/work/scratch/rz97hoku/ReinforcementLearning/tmp')
    ENV_path = get_dir(os.path.join(log_path, args.env))
    log_dir = os.path.join(ENV_path, datetime.datetime.now().strftime("ppo2-long-10ep-ent001-%m-%d-%H-%M-%S"))
    logger.configure(dir=log_dir)
    video_path = get_dir(logger.get_dir() + '/videos')
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, num_trials=5)


if __name__ == '__main__':
    main()
