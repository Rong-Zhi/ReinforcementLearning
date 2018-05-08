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

# path1 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-ent-10ep-5runs-05-05-13-09-58/progress.csv'
# path2 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-15ep-5runs-05-06-00-23-44/progress1.csv'
# path3 ='/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-15ep-ent-5runs-05-05-21-21-38/progress.csv'
# path4 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-20ep-5runs-05-05-21-26-21/progress.csv'
# path5 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-20ep-ent-5runs-05-05-21-26-47/progress.csv'
# path6 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-20ep-5runs-05-05-21-26-21/progress.csv'
# path7 = '/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-15ep-05-05-16-12-25/progress.csv'
path1 ='/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-10ep-ent-05-06-12-53-32/progress1.csv'
path2 ='/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-10ep-ent0001-05-06-20-18-02/progress.csv'
path3 ='/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-15ep-ent-05-06-20-18-30/progress.csv'
path4 ='/Users/zhirong/Documents/Masterthesis-code/tmp/LunarLanderContinuous-v2/ppo2-long-15ep-ent0001-05-06-20-18-56/progress.csv'

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes = axes.flatten()


def plots(i):
    for ax in axes:
        ax.clear()
    d1 = pd.read_csv(path1)
    d2= pd.read_csv(path2)
    d3 = pd.read_csv(path3)
    d4 = pd.read_csv(path4)
    # d5 = pd.read_csv(path5)
    # d6 = pd.read_csv(path6)
    # data = pd.concat([d3, d5])
    # data = d3
    data = pd.concat([d1, d2, d3, d4])

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