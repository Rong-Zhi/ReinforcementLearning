import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

sns.set(color_codes=True)
#
# path3= '/home/zhi/Documents/ReinforcementLearning/tmp/LunarLanderContinuous-v2/ppo2-ent2-05-03-18-41-39/progress.csv'
# path2 = '/home/zhi/Documents/ReinforcementLearning/tmp/LunarLanderContinuous-v2/ppo2-ent1-05-03-18-43-10/progress.csv'
# path1 = '/home/zhi/Documents/ReinforcementLearning/tmp/LunarLanderContinuous-v2/ppo2-ent-constant-05-05-11-26-59/progress.csv'
# path6 = '/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppofullentretrace05001-5000-04-23-15-29-36/progress.csv'


path3 ='/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-ent-constant-05-05-11-26-59/progress.csv'
# path1 = '/Users/zhirong/Documents/Masterthesis-code/tmp/BipedalWalker-v2/ppo-05-02-01-07-45/progress.csv'
path2 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-05-05-12-01-11/progress.csv'
path1 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-5ep-05-05-12-01-38/progress.csv'
# path4 = '/Users/zhirong/Documents/Masterthesis-code/tmp/RockSample/ppoent5-5000-04-13-02-09-29/progress.csv'


fig, axes = plt.subplots(1, 2, figsize=(14,5))
axes = axes.flatten()


def plots(i):
    for ax in axes:
        ax.clear()
    d1 = pd.read_csv(path1)
    d2= pd.read_csv(path2)
    d3 = pd.read_csv(path3)
    # d3=d3[:781]
    # d4 = pd.read_csv(path4)
    # d5 = pd.read_csv(path5)
    # d6 = pd.read_csv(path6)
    # data = pd.concat([d3, d5])
    # data = d1
    data = pd.concat([d1, d2, d3])

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