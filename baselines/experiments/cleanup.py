import pandas as pd
import glob

path = '/Users/zhirong/Documents/share/LunarLanderContinuousPOMDP-v0/guidedcopos-hist4-net32-new-{}-08-22-12-55'

dt = []

for i in range(10):
    # print(i)
    d = pd.read_csv(path.format(i)+'/progress.csv')
    d['trial'] = i
    # d.to_csv(path.format(i)+'/progress.csv', index=False)
    dt.append(d)
    # d.append(pd.read_csv(path.format(i)+'/progress.csv'))
data = pd.concat(dt)
data.to_csv(path.format(0)+'/progress1.csv')
