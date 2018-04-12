#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common.cmd_util import make_rocksample_env, rocksample_arg_parser
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
# from baselines.trpo_mpi import trpo_rocksample
from baselines.trpo_mpi import trpo_guided, trpo_rocksample, ppo_entropy_constraint
import os
import datetime
import tensorflow as tf


def train(num_timesteps, seed, num_trials=5):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)
    for i_trial in range(num_trials):
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        env = make_rocksample_env(workerseed, map_name="5x7", observation_type="field_vision_full_pos",
                                  observation_noise=True, n_steps=15)

        # genv = make_rocksample_env(workerseed, map_name="5x7", observation_type="fully_observable",
        #                           observation_noise=False, n_steps=15)

        trpo_rocksample.learn(env, policy_fn, timesteps_per_batch=5000, max_kl=0.01, cg_iters=10, cg_damping=0.1,
            max_iters=600, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, i_trial=i_trial)

        # ppo_entropy_constraint.learn(env, policy_fn,timesteps_per_batch=2048, max_kl=0.05,
        #           max_timesteps=num_timesteps,cg_iters=20, gamma=0.99, lam=0.95, entcoeff=0.0, cg_damping=0.1,
        #           vf_stepsize=1e-3, vf_iters=5, clip_param=0.2, schedule='linear', i_trial=i_trial)
        env.close()

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def main():
    # args = mujoco_arg_parser().parse_args()
    args = rocksample_arg_parser().parse_args()
    args.seed = 0
    log_path = get_dir("/Users/zhirong/Documents/Masterthesis-code/tmp")
    # log_path = get_dir("/home/zhi/Documents/ReinforcementLearning/tmp")
    ENV_path = get_dir(os.path.join(log_path, args.env))
    log_dir = os.path.join(ENV_path, datetime.datetime.now().strftime("trpoent5-5000-%m-%d-%H-%M-%S"))
    logger.configure(dir=log_dir)
    # train(num_timesteps=args.num_timesteps, seed=args.seed)
    train(num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()

