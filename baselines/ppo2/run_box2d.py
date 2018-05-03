#!/usr/bin/env python3
import argparse
# from baselines.common.cmd_util import mujoco_arg_parser
from baselines.common.cmd_util import control_arg_parser, make_control_env
from baselines import bench, logger
import os
import datetime

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import gym
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def train(env_id, num_timesteps, seed, num_trials):
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)

    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    set_global_seeds(seed)
    for i_trial in range(num_trials):
        env = DummyVecEnv([make_env])
        env = VecNormalize(env)
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:
            policy = MlpPolicy
            ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
                lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                ent_coef=0.5,
                lr=3e-4,
                cliprange=0.2,
                total_timesteps=num_timesteps, useentr=False, i_trial=i_trial)

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def main():
    # args = mujoco_arg_parser().parse_args()
    args = control_arg_parser().parse_args()
    args.seed = 0
    ntrial = 1
    # log_path = get_dir("/Users/zhirong/Documents/Masterthesis-code/tmp")
    log_path = get_dir("/home/zhi/Documents/ReinforcementLearning/tmp")
    ENV_path = get_dir(os.path.join(log_path, args.env))
    log_dir = os.path.join(ENV_path, datetime.datetime.now().strftime("ppo2-awake-%m-%d-%H-%M-%S"))
    logger.configure(dir=log_dir)
    video_path = get_dir(logger.get_dir() + '/videos')
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, num_trials=1)


if __name__ == '__main__':
    main()
