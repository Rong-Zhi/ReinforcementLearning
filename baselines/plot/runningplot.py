import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

path ="/home/zhi/Documents/ReinforcementLearning/tmp/Pendulum-v0/trpo-04-05-18-30-49/progress.csv"

fig, axes = plt.subplots(2,2, figsize=(10,5))
axes = axes.flatten()

def plots(i):
    for ax in axes:
        ax.clear()
    data = pd.read_csv(path)
    sns.tsplot(data=data, time='Iteration', value='EpRewMean', unit='trial', ax=axes[0])
    sns.tsplot(data=data, time='Iteration', value='entropy', unit='trial', ax=axes[1])

    sns.tsplot(data=data, time='Iteration', value='GEpRewMean', unit='trial', ax=axes[2])
    sns.tsplot(data=data, time='Iteration', value='gentropy', unit='trial', ax=axes[3])
    # x = np.random.rand(1000)
    # axes[0].plot(x)


if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, plots, interval=1000)
    # plots()
    plt.show()