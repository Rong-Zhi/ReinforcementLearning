import gym
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder

env = gym.make("CartPole-v1")
rec = VideoRecorder(env)
env.reset()
rec.capture_frame()
rec.close()
assert not rec.empty
assert not rec.broken
assert os.path.exists(rec.path)
f=open(rec.path)
assert  os.fstat(f.fileno()).st_size>100
