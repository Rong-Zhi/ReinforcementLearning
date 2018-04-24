import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

sns.set(color_codes=True)
#
path1 ="/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppoentretrace-full-5000-05001-04-23-14-18-18/progress.csv"
path2 ="/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppofullent-5000-05001-04-23-15-27-03/progress.csv"
# path3 = '/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppofullentlinear-5000-04-23-15-27-38/progress.csv'
# path4 = '/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppofullentlineargamma095-5000-04-23-15-28-01/progress.csv'
# path5 = '/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppofullentretracelinear-5000-04-23-15-29-04/progress.csv'
# path6 = '/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppofullentretrace05001-5000-04-23-15-29-36/progress.csv'

# path1 ='/Users/zhirong/Documents/Masterthesis-code/tmp/RockSample/ppoguidedentretrace-5000-04-23-22-37-34/progress.csv'
# path1 = '/Users/zhirong/Documents/Masterthesis-code/tmp/CartPole-v0/ppo-04-24-11-47-00/progress.csv'
# path2 = '/Users/zhirong/Documents/Masterthesis-code/tmp/RockSample/trpoent5-5000-04-12-22-31-23/progress.csv'
# path3 = '/Users/zhirong/Documents/Masterthesis-code/tmp/RockSample/ppo5-5000-04-13-02-08-29/progress.csv'
# path4 = '/Users/zhirong/Documents/Masterthesis-code/tmp/RockSample/ppoent5-5000-04-13-02-09-29/progress.csv'


fig, axes = plt.subplots(1, 2, figsize=(18,5))
axes = axes.flatten()

def plots(i):
    for ax in axes:
        ax.clear()
    d1 = pd.read_csv(path1)
    d2= pd.read_csv(path2)
    d1['Name']='PPO Partial observable'
    d2['Name']='PPO Fully observable'
    # d3 = pd.read_csv(path3)
    # d4 = pd.read_csv(path4)
    # d5 = pd.read_csv(path5)
    # d6 = pd.read_csv(path6)
    # data = pd.concat([d3, d5])
    # data = d3
    data = pd.concat([d1,d2])

    sns.tsplot(data=data, time='Iteration', value='EpRewMean',unit='trial', condition='Name', ax=axes[0])
    # sns.tsplot(data=d1, time='Iteration', value='EpDRewMean',unit='trial', condition='Name', ax=axes[1])
    sns.tsplot(data=data, time='Iteration', value='loss_ent', unit='trial', condition='Name',ax=axes[1])

    # sns.tsplot(data=d1, time='Iteration', value='GEpRewMean',unit='trial', condition='Name', ax=axes[2])
    # # sns.tsplot(data=d1, time='Iteration', value='GEpDRewMean',unit='trial', condition='Name', ax=axes[4])
    # sns.tsplot(data=d1, time='Iteration', value='gloss_gent', unit='trial', condition='Name',ax=axes[3])
    # axes[2].set_ylim([-2, 3])



if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, plots, interval=1000)
    # plots()
    plt.show()