import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

sns.set(color_codes=True)

path = '/home/zhi/Documents/share/LunarLanderContinuous-v2/ppo1-ent-dynamic002001-try-0/progress1.csv'


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
    d1 = pd.read_csv(path)
    # data = pd.concat([d0,d00, d1,d2, d3])
    data = d1
    # data = pd.concat([d1, d2, d3, d5, d6])

    sns.tsplot(data=data, time='Iteration', value='EpRewMean',unit='trial', condition='Name', ax=axes[0], ci=95)
    sns.tsplot(data=data, time='Iteration', value='entropy', unit='trial', condition='Name',ax=axes[1], ci=95)


if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, plots, interval=1000)
    # plots()
    plt.show()