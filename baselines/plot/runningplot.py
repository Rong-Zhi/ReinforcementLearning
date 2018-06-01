import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

sns.set(color_codes=True)

# path0 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-md-f163-hist16-bh9-batch32-0-05-28-23-56/progress1.csv'
# path1 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-md-f163-hist16-bh5-batch32-0-05-29-00-02/progress1.csv'
# path2 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-md-f1633-hist16-bh5-batch32-0-05-29-00-04/progress1.csv'
# path3 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-md-f1633-hist16-bh9-batch32-0-05-29-00-07/progress1.csv'
# path4 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-md-f1633-hist16-bh17-batch32-0-05-29-00-20/progress1.csv'
# path5 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-md-f163-hist16-bh17-batch32-0-05-28-23-47/progress1.csv'
# path1 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/clip-3l-0-05-21-19-46/progress1.csv'

path2 = '/home/zhi/Documents/ReinforcementLearning/tmp/LunarLanderContinuous-v2/copos-try-0-06-01-12-46/progress.csv'
path1 = '/home/zhi/Documents/ReinforcementLearning/tmp/LunarLanderContinuous-v2/ppo-md-try-0-06-01-13-15/progress.csv'
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes = axes.flatten()

# fig = plt.figure(1)

def plots(i):
    for ax in axes:
        ax.clear()
    # d0 = pd.read_csv(path0)
    d1 = pd.read_csv(path1)
    d2 = pd.read_csv(path2)
    # d3 = pd.read_csv(path3)
    # d4 = pd.read_csv(path4)
    # d5 = pd.read_csv(path5)

    data = pd.concat([d1, d2])
    # data = d1
    # data = pd.concat([d1, d2, d3, d4, d5, d6])

    sns.tsplot(data=data, time='Iteration', value='EpRewMean',unit='trial', condition='Name', ax=axes[0], ci=95)
    # sns.tsplot(data=d1, time='Iteration', value='EpDRewMean',unit='trial', condition='Name', ax=axes[1])
    sns.tsplot(data=data, time='Iteration', value='entropy', unit='trial', condition='Name',ax=axes[1], ci=95)

    # sns.tsplot(data=d1, time='Iteration', value='GEpRewMean',unit='trial', condition='Name', ax=axes[2])
    # # sns.tsplot(data=d1, time='Iteration', value='GEpDRewMean',unit='trial', condition='Name', ax=axes[4])
    # sns.tsplot(data=d1, time='Iteration', value='gloss_gent', unit='trial', condition='Name',ax=axes[3])
    # axes[2].set_ylim([-2, 3])



if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, plots, interval=1000)
    # plots()
    plt.show()