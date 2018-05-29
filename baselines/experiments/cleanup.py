import pandas as pd
import glob

path = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-md-f163-hist8-bh17-batch32-{}-05-28-23-42'

d = []
for i in range(10):
    d.append(pd.read_csv(path.format(i)+'/progress.csv'))
data = pd.concat(d)
data.to_csv(path.format(0)+'/progress1.csv')
