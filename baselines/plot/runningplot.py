import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

sns.set(color_codes=True)
#
# path1= '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/clip-3l-hist0-bh17-0-05-22-01-22/progress.csv'
# path2 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/clip-3l-hist10-bh17-4-05-22-01-22/progress1.csv'
# path3 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/clip-3l-hist20-bh17-0-05-22-01-23/progress1.csv'
# path4= '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-2l-hist0-bh5-0-05-22-01-19/progress1.csv'
# path5 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-2l-hist0-bh9-0-05-22-01-17/progress1.csv'
# path6 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-2l-hist0-bh17-0-05-22-01-12/progress1.csv'
# path7 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-2l-hist10-bh9-0-05-22-01-18/progress1.csv'
# path8 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-2l-hist10-bh17-0-05-22-01-15/progress1.csv'
# path9 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-3l-hist10-bh5-0-05-22-01-20/progress1.csv'
# path10 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-2l-hist20-bh5-0-05-22-01-18/progress1.csv'
# path11 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-2l-hist20-bh9-0-05-22-01-18/progress1.csv'
# path12 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/ent-2l-hist20-bh17-0-05-22-01-16/progress1.csv'

# path1 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/clip-3l-0-05-21-19-46/progress1.csv'
# path2 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-2l-0-05-21-19-45/progress1.csv'
# path3 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/ent-3l-0-05-21-19-36/progress1.csv'
# path4 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-20ep-5runs-05-05-21-26-21/progress.csv'
# path5 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-20ep-ent-5runs-05-05-21-26-47/progress.csv'
# path6 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-20ep-5runs-05-05-21-26-21/progress.csv'
# path7 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-15ep-05-05-16-12-25/progress.csv'
# path1 ='/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-10ep-ent-05-06-12-53-32/progress1.csv'
# path2 ='/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-10ep-ent0001-05-06-20-18-02/progress.csv'
# path3 ='/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-15ep-ent-05-06-20-18-30/progress.csv'
# path4 ='/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-15ep-ent0001-05-06-20-18-56/progress.csv'

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes = axes.flatten()


def plots(i):
    for ax in axes:
        ax.clear()
    # d1 = pd.read_csv(path1)
    # d2 = pd.read_csv(path2)
    # d3 = pd.read_csv(path3)
    # d4 = pd.read_csv(path4)
    # d5 = pd.read_csv(path5)
    # d6 = pd.read_csv(path6)
    # d7 = pd.read_csv(path7)
    # d8 = pd.read_csv(path8)
    # d9 = pd.read_csv(path9)
    # d10 = pd.read_csv(path10)
    # d11 = pd.read_csv(path11)
    # d12 = pd.read_csv(path12)
    # data = pd.concat([d10, d11, d12])
    # data = d7
    # data = pd.concat([d1, d2, d3, d4, d5, d6])

    sns.tsplot(data=data, time='Iteration', value='EpRewMean',unit='trial', condition='Name', ax=axes[0], ci=95)
    # sns.tsplot(data=d1, time='Iteration', value='EpDRewMean',unit='trial', condition='Name', ax=axes[1])
    sns.tsplot(data=data, time='Iteration', value='loss_ent', unit='trial', condition='Name',ax=axes[1], ci=95)

    # sns.tsplot(data=d1, time='Iteration', value='GEpRewMean',unit='trial', condition='Name', ax=axes[2])
    # # sns.tsplot(data=d1, time='Iteration', value='GEpDRewMean',unit='trial', condition='Name', ax=axes[4])
    # sns.tsplot(data=d1, time='Iteration', value='gloss_gent', unit='trial', condition='Name',ax=axes[3])
    # axes[2].set_ylim([-2, 3])



if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, plots, interval=1000)
    # plots()
    plt.show()