import pandas as pd

path1 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/clip-3l-0-05-21-14-18/progress.csv'
path2 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/clip-3l-1-05-21-14-18/progress.csv'
path3 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/clip-3l-2-05-21-14-18/progress.csv'
path4 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/clip-3l-3-05-21-14-18/progress.csv'
path5 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/clip-3l-4-05-21-14-19/progress.csv'


# path1 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-0-05-15-19-32/progress.csv'
# path2 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-1-05-15-19-33/progress.csv'
# path3 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-2-05-15-19-33/progress.csv'
# path4 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-4-05-15-19-33/progress.csv'
# path5 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-4-05-15-19-33/progress.csv'

d1 = pd.read_csv(path1)
d2 = pd.read_csv(path2)
d3 = pd.read_csv(path3)
d4 = pd.read_csv(path4)
d5 = pd.read_csv(path5)

data = pd.concat([d1, d2, d3, d4, d5])

data.to_csv('/Users/zhirong/Documents/share/LunarLanderContinuous-v2/clip-3l-1-05-21-14-18/progress1.csv')
