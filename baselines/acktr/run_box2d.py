#!/usr/bin/env python3
import sys
sys.path.append('/work/scratch/rz97hoku/ReinforcementLearning/')
# sys.path.append('/home/zhi/Documents/ReinforcementLearning/')
# sys.path.append('/Users/zhirong/Documents/ReinforcementLearning/')
import tensorflow as tf
from baselines import logger
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common.cmd_util import make_control_env, control_arg_parser
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
import os
import datetime



def train(env_id, num_timesteps, seed):
    env = make_control_env(env_id, seed, hist_len=None)

    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=2048,
            desired_kl=0.002,
            num_timesteps=num_timesteps, animate=False)

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
    ENV_path = get_dir(os.path.join(args.log_dir, args.env))
    log_dir = os.path.join(ENV_path, args.method + "-" +
                           '{}'.format(args.seed)) + "-" + \
              datetime.datetime.now().strftime("%m-%d-%H-%M")
    logger.configure(dir=log_dir)
    # logger.log("This is rank {}".format(rank))
    save_args(args)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == "__main__":
    main()
