import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

sns.set(color_codes=True)

path1 ="/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/trpo/progress.csv"
path2 ="/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppo-04-12-17-01-47/progress.csv"
path3 = '/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppoent-04-12-17-52-57/progress.csv'
# path = "/Users/zhirong/Documents/Masterthesis-code/tmp/RockSample/ppoconstraint-04-09-23-08-36/progress.csv"
# path = "/Users/zhirong/Documents/Masterthesis-code/tmp/Pendulum-v0/trpo-04-06-12-50-42/progress.csv"
fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes = axes.flatten()

def plots(i):
    for ax in axes:
        ax.clear()
    d1 = pd.read_csv(path1)
    d1['name']= 'trpo'
    d2= pd.read_csv(path2)
    d2['name'] = 'ppo'
    d3 = pd.read_csv(path3)
    d3['name'] = 'ppoent'
    data = pd.concat([d1[:600], d2[:600], d3])

    sns.tsplot(data=data, time='Iteration', value='EpRewMean',unit='trial', condition='name', ax=axes[0])
    sns.tsplot(data=data, time='Iteration', value='loss_ent', unit='trial', condition='name',ax=axes[1])

    # sns.tsplot(data=data, time='Iteration', value='GEpRewMean', unit='trial', ax=axes[2])
    # axes[2].set_ylim([-2, 3])
    # sns.tsplot(data=data, time='Iteration', value='gloss_gent', unit='trial', ax=axes[3])
    # x = np.random.rand(1000)
    # axes[0].plot(x)


if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, plots, interval=1000)
    # plots()
    plt.show()