    #!/usr/bin/env python3
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import tensorflow as tf
import numpy as np
import gym, logging
from baselines import logger
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import atari_arg_parser

def train(env_id, num_timesteps, seed, give_state, vf_iters, trial, nsteps, method, hist_len):
    from baselines.guidedcarla.nosharing_cnn_policy import CnnPolicy
    from baselines.guidedcarla import copos_mpi
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    # if rank == 0:
    #     logger.configure()
    # else:
    #     logger.configure(format_strs=[])

    workerseed = seed * 10000
    set_global_seeds(workerseed)

    #TODO:change the environment to carla
    env = make_atari(env_id)

    def policy_fn(name, ob_space, ac_space, ob_name, hist_len): #pylint: disable=W0613
        return CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, ob_name=ob_name, hist_len=hist_len)

    #TODO: check if monitor can deal with carla
    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    #TODO: check wrap deepmind and carla
    env = wrap_deepmind(env)
    env.seed(workerseed)

    timesteps_per_batch=nsteps
    beta = -1
    if beta < 0:
        nr_episodes = num_timesteps // timesteps_per_batch
        # Automatically compute beta based on initial entropy and number of iterations
        tmp_pi = policy_fn("tmp_pi", env.observation_space, env.action_space, ob_name="tmp_ob", hist_len=hist_len)
        sess.run(tf.global_variables_initializer())

        tmp_ob = np.zeros((1,) + env.observation_space.shape)
        entropy = sess.run(tmp_pi.pd.entropy(), feed_dict={tmp_pi.ob: tmp_ob})
        beta = 2 * entropy / nr_episodes
        print("Initial entropy: " + str(entropy) + ", episodes: " + str(nr_episodes))
        print("Automatically set beta: " + str(beta))

    copos_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, epsilon=0.01, beta=beta,
                    cg_iters=10, cg_damping=0.1, method=method,
                    max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=vf_iters, vf_stepsize=1e-3,
                    trial=trial, crosskl_coeff=0.01, kl_target=0.01, sess=sess)
    env.close()

def main():
    args = atari_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == "__main__":
    main()
