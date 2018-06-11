import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

sns.set(color_codes=True)

# path1 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/trop-0/progress1.csv'
# path2 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/COPOS-0-06-01-17-18/progress1.csv'
# path3 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ACKTR-0-06-03-13-37/progress1.csv'
# # path4 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ppo1-0/progress1.csv'
# path5 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ppo2-0/progress1.csv'
# path6 = '/Users/zhirong/Documents/share/clusterresults/LunarLanderContinuous-v2/ent-dynammic01-2l-0-05-23-11-51/progress1.csv'

# path1 = '/Users/zhirong/Documents/share/complete/ACKTR-hist16-0-06-04-11-44/progress1.csv'
# path3 = '/Users/zhirong/Documents/share/complete/COPOS-hist16-0-06-04-01-00/progress1.csv'
# path1 = '/Users/zhirong/Documents/share/complete/COPOS-hist4-0-06-04-12-53/progress1.csv'
# path2 = '/Users/zhirong/Documents/share/complete/COPOS-hist8-0-06-04-12-51/progress1.csv'
# path0 = '/Users/zhirong/Documents/share/complete/COPOS-hist1-0-06-05-19-15/progress1.csv'
# path00 = '/Users/zhirong/Documents/share/complete/COPOS-hist2-0-06-05-19-15/progress1.csv'
# path3 = '/Users/zhirong/Documents/share/complete/ppo2-hist16-0/progress1.csv'
# path4 = '/Users/zhirong/Documents/share/complete/trpo-hist16-0/progress1.csv'
# path5 = '/Users/zhirong/Documents/share/complete/ppo2-hist16-ent001-0/progress1.csv'
# path6 = '/Users/zhirong/Documents/share/complete/ppo2-hist16-ent002-0/progress1.csv'
# path7 = '/Users/zhirong/Documents/share/complete/ppo2-hist16-ent004-0/progress1.csv'
# path2 = '/home/zhi/Documents/ReinforcementLearning/tmp/LunarLanderContinuous-v2/copos-try-0-06-01-12-46/progress.csv'
# path1 = '/home/zhi/Documents/ReinforcementLearning/tmp/LunarLanderContinuous-v2/ppo-md-try-0-06-01-13-15/progress.csv'

# path = '/Users/zhirong/Documents/ReinforcementLearning/tmp/LunarLanderContinuous-v2/ppo-entropy-try-0-06-10-17-07/progress.csv'
path1 = '/Users/zhirong/Documents/ReinforcementLearning/tmp/LunarLanderContinuous-v2/ppo-entropy002-try-0-06-10-17-32/progress.csv'
# path5 = '/Users/zhirong/Documents/share/clusterresults/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist16-bh17-batch32-0-05-24-15-51/progress1.csv'
# path1 = '/Users/zhirong/Documents/share/clusterresults/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist1-bh17-batch32-0-05-24-15-50/progress1.csv'
# path2 = '/Users/zhirong/Documents/share/clusterresults/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist2-bh17-batch32-0-05-24-15-50/progress1.csv'
# path3 = '/Users/zhirong/Documents/share/clusterresults/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist4-bh17-batch32-0-05-24-15-50/progress1.csv'
# path4 = '/Users/zhirong/Documents/share/clusterresults/LunarLanderContinuousPOMDP-v0/ent-dynammic01-2l-hist8-bh17-batch32-0-05-24-15-50/progress1.csv'

# path6 = '/Users/zhirong/Documents/share/clusterresults/LunarLanderContinuousPOMDP-v0/ent-md-f1633-hist8-bh17-batch32-0-05-29-00-20/progress1.csv'
# path7 = '/Users/zhirong/Documents/share/clusterresults/LunarLanderContinuousPOMDP-v0/ent-md-f1633-hist16-bh17-batch32-0-05-29-00-20/progress1.csv'


fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes = axes.flatten()

# fig = plt.figure(1)

def plots(i):
    for ax in axes:
        ax.clear()


    # d1 = pd.read_csv(path1)
    # d2 = pd.read_csv(path2)
    # d3 = pd.read_csv(path3)
    # d3['Name'] = 'COPOS-hist16'
    # d4 = pd.read_csv(path4)
    # d5 = pd.read_csv(path5)
    # d6 = pd.read_csv(path6)
    # d7 = pd.read_csv(path7)
    # d = pd.read_csv(path)
    d1 = pd.read_csv(path1)
    # data = pd.concat([d0,d00, d1,d2, d3])
    data = d1
    # data = pd.concat([d1, d2, d3, d5, d6])

    sns.tsplot(data=data, time='Iteration', value='EpRewMean',unit='trial', condition='Name', ax=axes[0], ci=95)
    # sns.tsplot(data=data, time='Iteration', value='EpRewMean', unit='trial', condition='Name', ax=axes[0], ci=95, err_style='unit_traces')
    # sns.tsplot(data=d1, time='Iteration', value='EpDRewMean',unit='trial', condition='Name', ax=axes[1])z
    sns.tsplot(data=data, time='Iteration', value='loss_ent', unit='trial', condition='Name',ax=axes[1], ci=95)


if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, plots, interval=1000)
    # plots()
    plt.show()