import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

sns.set(color_codes=True)
#
# path1 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/COPOS-10seeds-hist1-0-06-11-12-23/progress1.csv'
# path2 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/COPOS-10seeds-hist2-0-06-11-12-24/progress1.csv'
# path3 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/COPOS-10seeds-hist4-0-06-11-12-24/progress1.csv'
# path4 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/COPOS-10seeds-hist8-0-06-11-12-25/progress1.csv'
# path5 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/COPOS-10seeds-hist16-0-06-11-12-25/progress1.csv'
#
# path6 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/COPOS-10seeds-64neuron-hist8-0-06-11-14-28/progress1.csv'
# path7 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/COPOS-10seeds-64neuron-hist16-0-06-11-14-28/progress1.csv'
# path8 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/COPOS-10seeds-64neuron-hist1-0-06-12-12-16/progress1.csv'
# path9 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/COPOS-10seeds-64neuron-hist2-0-06-12-12-12/progress1.csv'
# path10 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/COPOS-10seeds-64neuron-hist4-0-06-12-12-10/progress1.csv'
#
# path ='/home/zhi/Documents/share/LunarLanderContinuous-v2/ppo1-ent-dynamic001001-try-0/progress1.csv'
#
# path11 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/PPO2-10seeds-64neuron-hist1-0-06-12-12-03/progress1.csv'
# path12 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/PPO2-10seeds-64neuron-hist2-0-06-12-12-02/progress1.csv'
# path13 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/PPO2-10seeds-64neuron-hist4-0-06-12-12-02/progress1.csv'
# path14 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/PPO2-10seeds-64neuron-hist8-0-06-12-12-02/progress1.csv'
# path15 = '/home/zhi/Documents/share/LunarLanderContinuousPOMDP-v0/PPO2-10seeds-64neuron-hist16-0-06-12-12-00/progress1.csv'


# path1 = '/Users/zhirong/Documents/share/copos64neuron/COPOS-10seeds-64neuron-hist1-0-06-12-12-14/progress1.csv'
# path2 = '/Users/zhirong/Documents/share/copos64neuron/COPOS-10seeds-64neuron-hist2-0-06-12-12-11/progress1.csv'
# path3 = '/Users/zhirong/Documents/share/copos64neuron/COPOS-10seeds-64neuron-hist4-0-06-12-12-10/progress1.csv'
# path4 = '/Users/zhirong/Documents/share/copos64neuron/COPOS-10seeds-64neuron-hist8-0-06-17-13-52/progress1.csv'
# path5 = '/Users/zhirong/Documents/share/copos64neuron/COPOS-10seeds-64neuron-hist16-0-06-17-13-52/progress1.csv'

path1 = '/Users/zhirong/Documents/share/guided/guided-ratio-diffinput-hist4-net32-epoch5-0-07-04-00-56/progress1.csv'
path2 = '/Users/zhirong/Documents/share/guided/guided-ratio-diffinput-hist4-net32-epoch10-0-07-04-00-56/progress1.csv'
path3 = '/Users/zhirong/Documents/share/guided/guided-ratio-diffinput-hist4-net64-epoch5-0-07-04-00-57/progress1.csv'
path4 = '/Users/zhirong/Documents/share/guided/guided-ratio-diffinput-hist4-net64-epoch10-0-07-04-00-59/progress1.csv'
path5 = '/Users/zhirong/Documents/share/guided/guided-diffinput-hist4-net32-epoch5-0-07-03-17-46/progress1.csv'
path6 = '/Users/zhirong/Documents/share/guided/guided-diffinput-hist4-net32-epoch10-0-07-04-00-51/progress1.csv'
path7 = '/Users/zhirong/Documents/share/guided/guided-diffinput-hist4-net64-epoch5-0-07-03-17-48/progress1.csv'
path8 = '/Users/zhirong/Documents/share/guided/guided-diffinput-hist4-net64-epoch10-0-07-03-17-48/progress1.csv'
path9 = '/Users/zhirong/Documents/share/complete/COPOS-10seeds-hist4-0-06-11-12-24/progress1.csv'
path10 = '/Users/zhirong/Documents/share/complete/COPOS-10seeds-64neuron-hist4-0-06-12-12-10/progress1.csv'

path = '/Users/zhirong/Documents/share/LunarLanderContinuousPOMDP-v0/log-hist4-net32-0-07-04-17-17/progress1.csv'



fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes = axes.flatten()

# fig = plt.figure(1)

def plots(i):
    for ax in axes:
        ax.clear()

    d = pd.read_csv(path)
    # d1 = pd.read_csv(path1)
    # d1['Name']='guidedcopos-hist4-net32'
    # # d2 = pd.read_csv(path2)
    # # d2['Name'] = 'guidedcpos-epoch10-hist4-net32'
    # d3 = pd.read_csv(path3)
    # d3['Name'] = 'guidedcopos-hist4-net64'
    #
    # d9 = pd.read_csv(path9)
    # d9['Name'] = 'copos-hist4-net32'
    #
    # d10 = pd.read_csv(path10)
    # d10['Name'] = 'copos-hist4-net64'

    # d4 = pd.read_csv(path4)
    # d4['Name'] = 'GuidedCOPOS-ratio-epoch10-net64'
    # d5 = pd.read_csv(path5)
    # d5['Name'] = 'guidedcopos-log-net32'
    # d6 = pd.read_csv(path6)
    # d6['Name'] = 'log-epoch10-net32'
    # d7 = pd.read_csv(path7)
    # d7['Name'] = 'guidedcopos-log-net64'
    # d8 = pd.read_csv(path8)
    # d8['Name'] = 'log-epoch10-net64'
    # d9 = pd.read_csv(path9)
    # d10 =pd.read_csv(path10)

    # d11 = pd.read_csv(path11)
    # d12 = pd.read_csv(path12)
    # d13 = pd.read_csv(path13)
    # d14 = pd.read_csv(path14)
    # d15 = pd.read_csv(path15)



    # d = pd.read_csv(path)
    # d1 = pd.read_csv(path)
    data = d
    # data = pd.concat([d1,d3,d9,d10])
    # data = pd.concat([d8, d9, d10, d6, d7])
    # rr_style = 'unit_traces'
    sns.tsplot(data=data, time='Iteration', value='EpRewMean',unit='trial', condition='Name', ax=axes[0], ci=95)
    sns.tsplot(data=data, time='Iteration', value='entropy', unit='trial', condition='Name',ax=axes[1], ci=95)
    # sns.tsplot(data=data, time='Iteration', value='meankl', unit='trial', condition='Name', ax=axes[2], ci=95)
    # sns.tsplot(data=data, time='Iteration', value='meancrosskl', unit='trial', condition='Name', ax=axes[3], ci=95)

    # sns.tsplot(data=data, time='Iteration', value='EpRewMean',unit='trial', condition='Name', ax=axes[4], ci=95)
    # sns.tsplot(data=data, time='Iteration', value='gentropy', unit='trial', condition='Name',ax=axes[5], ci=95)
    # sns.tsplot(data=data, time='Iteration', value='gmeankl', unit='trial', condition='Name', ax=axes[6], ci=95)
    # sns.tsplot(data=data, time='Iteration', value='gmeancrosskl', unit='trial', condition='Name', ax=axes[7], ci=95)

if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, plots, interval=1000)
    # plots()
    plt.show()