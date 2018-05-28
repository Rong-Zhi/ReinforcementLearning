import pandas as pd
import glob

path = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist0-bh17-batch32-{}-05-24-15-48'

d = []
for i in (2,4,7,8):
    d.append(pd.read_csv(path.format(i)+'/progress.csv'))
data = pd.concat(d)
data.to_csv(path.format(2)+'/progress1.csv')
