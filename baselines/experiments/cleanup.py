import pandas as pd
import glob

path = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/COPOS-10seeds-64neuron-hist4-{}-06-12-12-10'

dt = []

for i in range(10):
    # # print(i)
    d = pd.read_csv(path.format(i)+'/progress.csv')
    # d['trial'] = i
    # d.to_csv(path.format(i)+'/progress.csv', index=False)
    dt.append(d)
    # d.append(pd.read_csv(path.format(i)+'/progress.csv'))
data = pd.concat(dt)
data.to_csv(path.format(0)+'/progress1.csv')
