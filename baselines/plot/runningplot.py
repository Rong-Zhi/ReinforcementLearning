import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

sns.set(color_codes=True)
#
# path1= '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/clip-0-05-15-03-17/progress1.csv'
# path2 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/clip-hist10-0-05-15-19-27/progress1.csv'
# path3 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/clip-hist15-0-05-15-19-26/progress1.csv'
# path4= '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-0-05-15-03-12/progress1.csv'
# path5 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist10-0-05-15-19-30/progress1.csv'
# path6 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/entropy001-hist15-0-05-15-19-32/progress1.csv'

path1 = '/Users/zhirong/Documents/share/LunarLanderContinuous-v2/clip-3l-0-05-21-16-06/progress.csv'
path2 = '/Users/zhirong/Documents/ReinforcementLearning/tmp/LunarLanderContinuous-v2/ppo1-clip-2l-0-05-20-19-52/progress.csv'
path3 ='/Users/zhirong/Documents/ReinforcementLearning/tmp/LunarLanderContinuous-v2/ppo1-clip-3l-ent-0-05-20-21-31/progress.csv'
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
    d2 = pd.read_csv(path1)
    d3 = pd.read_csv(path2)
    d1 = pd.read_csv(path3)
    # d4 = pd.read_csv(path4)
    # d5 = pd.read_csv(path5)
    # d6 = pd.read_csv(path6)
    data = pd.concat([d1, d2, d3])
    # data = d1
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