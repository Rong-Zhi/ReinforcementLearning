import pandas as pd
import glob

path = '/Users/zhirong/Documents/share/copos64neuron/COPOS-10seeds-64neuron-hist16-{}-06-17-13-52'

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
