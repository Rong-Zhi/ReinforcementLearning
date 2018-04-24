#!/usr/bin/env python3

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser, make_control_env, control_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import os
import datetime


def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple, ppo_guided
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)
    # env = make_mujoco_env(env_id, seed)
    env = make_control_env(env_id, seed)
    i_trial = 1

    # genv = make_control_env(env_id, seed)
    #
    #
    # ppo_guided.learn(env, genv, i_trial, policy_fn,
    #         max_iters=100,
    #         timesteps_per_actorbatch=2048,
    #         clip_param=0.2, entp=0.5,
    #         optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
    #         gamma=0.99, lam=0.95, schedule='linear', useentr=False, retrace=False
    #                     )



    pposgd_simple.learn(env, i_trial, policy_fn,
            max_iters=100,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear'
        )
    env.close()

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def main():
    # args = mujoco_arg_parser().parse_args()
    args = control_arg_parser().parse_args()
    log_path = get_dir("/Users/zhirong/Documents/Masterthesis-code/tmp")
    # log_path = get_dir("/home/zhi/Documents/ReinforcementLearning/tmp")
    ENV_path = get_dir(os.path.join(log_path, args.env))
    log_dir = os.path.join(ENV_path, datetime.datetime.now().strftime("ppo-%m-%d-%H-%M-%S"))
    logger.configure(dir=log_dir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
