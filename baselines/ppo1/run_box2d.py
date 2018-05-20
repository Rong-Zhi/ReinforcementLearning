#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import sys
sys.path.append('/work/scratch/rz97hoku/ReinforcementLearning')
# sys.path.append('/home/zhi/Documents/ReinforcementLearning/')
# sys.path.append('/Users/zhirong/Documents/Masterthesis-code')
from baselines.common.cmd_util import make_control_env, control_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import os
import datetime
import os.path as osp
from baselines.ppo1 import mlp_policy, ppo_guided, pposgd_simple
from baselines.env.envsetting import newenv

def train(env_id, num_timesteps, seed, nsteps, batch_size, epoch,
        method, hist_len, net_size, i_trial, load_path, checkpoint,
          num_cpu, use_entr):

    U.make_session(num_cpu=num_cpu).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=net_size[0], num_hid_layers=len(net_size))
        # return mlp_policy.MlpBetaPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        #     hid_size=64, num_hid_layers=2)

    env = make_control_env(env_id, seed, hist_len)

    pposgd_simple.learn(env, i_trial=i_trial, policy_fn=policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=nsteps,
            clip_param=0.2, entp=0.01,
            optim_epochs=epoch, optim_stepsize=3e-4, optim_batchsize=batch_size,
            gamma=0.99, lam=0.95, schedule='linear', useentr=use_entr,
            load_path=load_path, method=method, usecheckpoint=checkpoint)
    env.close()

def render(env_id, seed, hist_len, net_size, load_path, video_path, iters):
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=net_size[0], num_hid_layers=len(net_size))

    env = make_control_env(env_id, seed, hist_len)
    pposgd_simple.render(policy=policy_fn, env=env, load_path=load_path, video_path=video_path, iters_so_far=iters)
    env.close()

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def save_args(args):
    for arg in vars(args):
        logger.log("{}: ".format(arg), getattr(args, arg))

def main():
    args = control_arg_parser().parse_args()
    if args.env == 'LunarLanderContinuousPOMDP-v0':
        newenv(hist_len=args.hist_len)

    if args.train:
        print(args.train)
        ENV_path = get_dir(os.path.join(args.log_dir, args.env))
        log_dir = os.path.join(ENV_path, args.method +"-"+
                               '{0}'.format(args.seed))+"-" +\
                  datetime.datetime.now().strftime("%m-%d-%H-%M")

        logger.configure(dir=log_dir)
        save_args(args)
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, nsteps=args.nsteps,
              batch_size=args.batch_size, epoch=args.epoch, method=args.method, hist_len=args.hist_len,
              net_size=args.net_size, i_trial=args.seed, load_path=args.load_path,
              checkpoint=args.checkpoint, num_cpu=args.ncpu, use_entr=int(args.use_entr))

    if args.render:
        video_path = osp.split(osp.split(args.load_path)[0])[0]
        render(args.env, net_size=args.net_size,
               load_path=args.load_path, seed=args.seed, iters=args.iters, video_path=video_path)

if __name__ == '__main__':
    main()

