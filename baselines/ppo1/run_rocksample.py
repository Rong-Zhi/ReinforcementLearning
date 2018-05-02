#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from baselines.common.cmd_util import make_rocksample_env, rocksample_arg_parser
from baselines.common.cmd_util import make_control_env, control_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import os
import datetime



def train(env_id, num_timesteps, seed, num_trials=5):
    from baselines.ppo1 import mlp_policy, ppo_guided, pporocksample, ppo_guided2, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
        # return mlp_policy.MlpBetaPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        #     hid_size=64, num_hid_layers=2)

    for i_trial in range(num_trials):

        env = make_control_env(env_id, seed)

        # normalized histroy stepsize 15 for field vision full position rocksample
        # env = make_rocksample_env(seed, map_name="5x7", observation_type="field_vision_full_pos",
                                  # observation_noise=True, n_steps=15)

        # normalized fully observable rocksample
        # env = make_rocksample_env(seed, map_name="5x7", observation_type="fully_observable",
        #                            observation_noise=False, n_steps=15)

        # # guided way of normalized fully observable rocksample with history timestep 15
        # genv = make_control_env(env_id, seed)
        #
        # ppo_guided.learn(env, genv, i_trial, policy_fn,
        #         max_iters=1000,
        #         timesteps_per_actorbatch=5000,
        #         clip_param=0.2, entp=0.5,
        #         optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=32,
        #         gamma=0.99, lam=0.95, schedule='linear', useentr=False, retrace=False
        #                     )

        # pposgd_simple.learn(env, i_trial, policy_fn,
        #         max_iters=1000,
        #         timesteps_per_actorbatch=2048,
        #         clip_param=0.2, entcoeff=0.5,
        #         optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=32,
        #         gamma=0.99, lam=0.95, schedule='linear')

        pporocksample.learn(env, i_trial, policy_fn,
                max_iters=800,
                timesteps_per_actorbatch=2048,
                clip_param=0.2, entp=0.3,
                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=32,
                gamma=0.99, lam=0.95, schedule='linear', useentr=True, retrace=False
                            )
        env.close()



def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def main():
    # args = mujoco_arg_parser().parse_args()
    args= control_arg_parser().parse_args()
    # args = rocksample_arg_parser().parse_args()
    args.seed = 0
    log_path = get_dir("/Users/zhirong/Documents/Masterthesis-code/tmp")
    # log_path = get_dir("/home/zhi/Documents/ReinforcementLearning/tmp")
    ENV_path = get_dir(os.path.join(log_path, args.env))
    log_dir = os.path.join(ENV_path, datetime.datetime.now().strftime("ppo1-%m-%d-%H-%M-%S"))
    logger.configure(dir=log_dir)
    video_path = get_dir(logger.get_dir()+'/videos')
    train(env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()

