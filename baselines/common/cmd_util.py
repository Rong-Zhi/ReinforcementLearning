"""
Helpers for scripts like run_atari.py.
"""

import os
import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from mpi4py import MPI

from baselines.env.lunar_lander_pomdp import LunarLanderContinuousPOMDP
from baselines.env.lunar_lander import LunarLanderContinuous

# from rllab.envs.normalized_env import normalize
# from rllab.envs.pomdp.rock_sample_env import RockSampleEnv
# from rllab.envs.history_env import HistoryEnv


def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_mujoco_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env

def make_control_env(env_id, seed, hist_len):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    if env_id == 'LunarLanderContinuousPOMDP-v0':
        env = LunarLanderContinuousPOMDP(hist_len=hist_len)
    else:
        env = gym.make(env_id)
    env = Monitor(env, logger.get_dir(), allow_early_resets=True)
    env.seed(seed)
    return env


# def make_rocksample_env(seed, map_name, observation_type, observation_noise, n_steps):
#     """
#      Create a wrapped, monitored (Field)rocksample environment
#      (without seed)
#     """
#     set_global_seeds(seed)
#     # env = normalize(HistoryEnv(RockSampleEnv(map_name=map_name, observation_type=observation_type,
#     #                observation_noise=observation_noise), n_steps=n_steps), scale_reward=1)
#     env = normalize(RockSampleEnv(map_name=map_name, observation_type=observation_type,
#                    observation_noise=observation_noise), scale_reward=1)
#     # env.seed(seed)
#     env = Monitor(env, logger.get_dir(), allow_early_resets=True)
#     return env


def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    return parser

def mujoco_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser

def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser


def control_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    # parser.add_argument('--log_dir',type=str, default='/Users/zhirong/Documents/Masterthesis-code/tmp')
    parser.add_argument('--log_dir',type=str, default='/work/scratch/rz97hoku/ReinforcementLearning/tmp')
    parser.add_argument('--env', help='environment ID', type=str, default='LunarLanderContinuousPOMDP-v0')
    parser.add_argument('--net_size', help='Network size', type=tuple, default=(64,64,64))
    parser.add_argument('--hist_len', help='History Length', type=int, default=0)
    parser.add_argument('--nsteps', help='timesteps each iteration', type=int, default=2048)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--epoch', help='epoch', type=int, default=15)
    parser.add_argument('--method', help='method', type=str, default='entropy001')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=int(2e6))
    parser.add_argument('--train', help='train', type=bool, default=True)
    parser.add_argument('--render', help='render', type=bool, default=False)
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--checkpoint', help='Use saved checkpoint?', type=bool, default=False)
    parser.add_argument('--iters', help='Iterations so far(to produce videos)', default=0)
    parser.add_argument('--use_entr', help='Use dynammic entropy regularization term?', type=bool, default=False)
    return parser

def rocksample_arg_parser():
    """
    Create an argparse.ArgumentParser for run_rocksample.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='RockSample')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser