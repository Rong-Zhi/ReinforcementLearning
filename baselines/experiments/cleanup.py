import pandas as pd
import glob

path = '/Users/zhirong/Documents/share/todo/COPOS-hist2-{}-06-05-19-15'

d = []

for i in range(5):
    # # print(i)
    # d = pd.read_csv(path.format(i)+'/progress.csv')
    # d['trial'] = i
    # d.to_csv(path.format(i)+'/progress.csv', index=False)

    d.append(pd.read_csv(path.format(i)+'/progress.csv'))
data = pd.concat(d)
data.to_csv(path.format(0)+'/progress1.csv')
