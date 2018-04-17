import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

sns.set(color_codes=True)

path1 ="/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppoguide-full-04-17-17-21-19/progress.csv"
path2 ="/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppo-04-12-17-01-47/progress.csv"
path3 = '/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/ppoent-04-12-17-52-57/progress.csv'
path4 = '/home/zhi/Documents/ReinforcementLearning/tmp/RockSample/trpoent5000-04-12-18-48-43/progress.csv'

# path1 ='/Users/zhirong/Documents/Masterthesis-code/tmp/RockSample/trpo5-5000-04-12-23-25-33/progress.csv'
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
    d3 = pd.read_csv(path3)
    d4 = pd.read_csv(path4)
    # data = pd.concat([d1, d2, d3, d4])
    data = d1
    sns.tsplot(data=data, time='Iteration', value='EpRewMean',unit='trial', condition='Name', ax=axes[0])
    sns.tsplot(data=data, time='Iteration', value='loss_ent', unit='trial', condition='Name',ax=axes[1])
    # plt.title('TRPO')
    # sns.tsplot(data=d2, time='Iteration', value='EpRewMean',unit='trial', condition='Name', ax=axes[2])
    # sns.tsplot(data=d2, time='Iteration', value='loss_ent', unit='trial', condition='Name',ax=axes[3])
    # plt.title('TRPOent')
    # sns.tsplot(data=d3, time='Iteration', value='EpRewMean',unit='trial', condition='Name', ax=axes[4])
    # sns.tsplot(data=d3, time='Iteration', value='loss_ent', unit='trial', condition='Name',ax=axes[5])
    # plt.title('PPO')
    # sns.tsplot(data=d4, time='Iteration', value='EpRewMean',unit='trial', condition='Name', ax=axes[6])
    # sns.tsplot(data=d4, time='Iteration', value='loss_ent', unit='trial', condition='Name',ax=axes[7], )
    # plt.title('PPOent')
    # sns.tsplot(data=data, time='Iteration', value='GEpRewMean', unit='trial', ax=axes[2])
    # axes[2].set_ylim([-2, 3])
    # sns.tsplot(data=data, time='Iteration', value='gloss_gent', unit='trial', ax=axes[3])
    # x = np.random.rand(1000)
    # axes[0].plot(x)


if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, plots, interval=1000)
    # plots()
    plt.show()