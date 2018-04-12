import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

path ="/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppoent-04-11-15-31-55/progress.csv"
# path = "/Users/zhirong/Documents/Masterthesis-code/tmp/RockSample/ppoconstraint-04-09-23-08-36/progress.csv"
# path = "/Users/zhirong/Documents/Masterthesis-code/tmp/Pendulum-v0/trpo-04-06-12-50-42/progress.csv"
fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes = axes.flatten()

def plots(i):
    for ax in axes:
        ax.clear()
    data = pd.read_csv(path)
    sns.tsplot(data=data, time='Iteration', value='EpRewMean', unit='trial', ax=axes[0])
    sns.tsplot(data=data, time='Iteration', value='loss_ent', unit='trial', ax=axes[1])
    #
    # sns.tsplot(data=data, time='Iteration', value='GEpRewMean', unit='trial', ax=axes[2])
    # axes[2].set_ylim([-2, 3])
    # sns.tsplot(data=data, time='Iteration', value='gloss_gent', unit='trial', ax=axes[3])
    # x = np.random.rand(1000)
    # axes[0].plot(x)


if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, plots, interval=1000)
    # plots()
    plt.show()