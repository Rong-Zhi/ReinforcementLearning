import pandas as pd

# path1 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-3l-0-05-21-19-36/progress.csv'
# path2 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-3l-1-05-21-19-36/progress.csv'
# path3 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-3l-2-05-21-19-36/progress.csv'
# path4 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-3l-3-05-21-19-36/progress.csv'
# path5 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-3l-4-05-21-19-36/progress.csv'


path1 = '/home/zhi/Documents/share/LunarLanderContinuous-v2/05-22-19-15-1/progress.csv'
path2 = '/home/zhi/Documents/share/LunarLanderContinuous-v205-22-19-15-2/progress.csv'
path3 = '/home/zhi/Documents/share/LunarLanderContinuous-v2/05-22-16-37-2/progress.csv'
path4 = '/home/zhi/Documents/share/LunarLanderContinuous-v2/05-22-16-37-3/progress.csv'
# path5 = '/home/zhi/Documents/share/LunarLanderContinuous-v2/05-22-16-52-4/progress.csv'

d1 = pd.read_csv(path1)
d2 = pd.read_csv(path2)
d3 = pd.read_csv(path3)
d4 = pd.read_csv(path4)


d1['trial'] = 0
d2['trial'] = 1
d3['trial'] = 2
d4['trial'] = 3

data = pd.concat([d1, d2, d3, d4])

data.to_csv('/home/zhi/Documents/share/LunarLanderContinuous-v2/05-22-16-37-0/progress1.csv')
