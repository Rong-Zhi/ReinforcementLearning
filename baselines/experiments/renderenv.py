import gym
import os
import pandas as pd
import imageio
import joblib
import tensorflow as tf

from baselines import bench, logger
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv




def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def get_dir(path):
    """
    Create a path
    """
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def env_arg_parser():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='LunarLanderContinuousPOMDP-v0')
    parser.add_argument('--hist_len', help='History Length(just for POMDP env)', type=int, default=0)
    parser.add_argument('--load_path',type=str, default='/home/zhi/Documents/ReinforcementLearning/tmp')
    parser.add_argument('--save_path',type=str, default='/home/zhi/Documents/ReinforcementLearning/tmp')
    parser.add_argument('--fps', type=int, default=20)
    return parser


def load(load_path):
    loaded_params = joblib.load(load_path)
    restores = []
    for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
    sess.run(restores)

def main():
    args = env_arg_parser().parse_args()
    load_path = args.load_path
    save_path = args.save_path
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(args.load_path)


    def make_env():
        if args.env == 'LunarLanderContinuousPOMDP-v0':
            from baselines.env.lunar_lander_pomdp import LunarLanderContinuousPOMDP
            env = LunarLanderContinuousPOMDP(hist_len=args.hist_len)
        else:
            env = gym.make(args.env)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)





    ob = env.reset()

    # d = {'Pos_x':[], 'Pos_y':[], 'Vel_x':[], 'Vel_y':[], 'Angle':[],
    #      'Ang_Vel':[], 'Touch_l':[], 'Touch_r':[], 'Block':[]}
    # observation = pd.DataFrame(data=d)

    frames = []
    while True:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        act = env.action_space.sample()
        ob, rwd, done, _ = env.step(act)
        # print(ob)
        if done:
            imageio.mimsave(save_path + '/' + 'example.gif', frames, fps=20)
            print('Save video')
            break



