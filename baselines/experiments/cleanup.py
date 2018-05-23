import pandas as pd

# path1 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-3l-0-05-21-19-36/progress.csv'
# path2 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-3l-1-05-21-19-36/progress.csv'
# path3 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-3l-2-05-21-19-36/progress.csv'
# path4 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-3l-3-05-21-19-36/progress.csv'
# path5 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-3l-4-05-21-19-36/progress.csv'


path1 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist0-bh9-batch32-0-05-23-15-49/progress.csv'
path2 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist0-bh9-batch32-1-05-23-15-49/progress.csv'
path3 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist0-bh9-batch32-2-05-23-15-49/progress.csv'
path4 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist0-bh9-batch32-3-05-23-15-49/progress.csv'
path5 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist0-bh9-batch32-4-05-23-15-49/progress.csv'

d1 = pd.read_csv(path1)
d2 = pd.read_csv(path2)
d3 = pd.read_csv(path3)
d4 = pd.read_csv(path4)
d5 = pd.read_csv(path5)


data = pd.concat([d3, d4, d5])

data.to_csv('/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist0-bh9-batch32-0-05-23-15-49/progress1.csv')
