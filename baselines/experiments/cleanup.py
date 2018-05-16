import pandas as pd

# path1 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-10ep-ent0001--05-06-13-55-41/progress.csv'
# path2 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-10ep-ent0001--05-06-13-55-47/progress.csv'
# path3 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-10ep-ent0001--05-06-13-55-54/progress.csv'
# path4 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-10ep-ent0001--05-06-13-55-58/progress.csv'
# path5 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-10ep-ent0001--05-06-13-56-02/progress.csv'


path1 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-0-05-15-19-32/progress.csv'
path2 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-1-05-15-19-33/progress.csv'
path3 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-2-05-15-19-33/progress.csv'
path4 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-4-05-15-19-33/progress.csv'
# path5 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-4-05-15-19-33/progress.csv'

d1 = pd.read_csv(path1)
d2 = pd.read_csv(path2)
d3 = pd.read_csv(path3)
d4 = pd.read_csv(path4)
# d5 = pd.read_csv(path5)

data = pd.concat([d1, d2, d3, d4])

data.to_csv('/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-0-05-15-19-32/progress1.csv')