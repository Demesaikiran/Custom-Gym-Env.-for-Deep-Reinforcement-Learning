import numpy as np
import os
import matplotlib.pyplot as plt

# Init. path
path = os.getcwd()

# Load all the data frames
sum_train = np.load('data/score_history.npy', allow_pickle=False)
avg_train = np.load('data/avg_history.npy', allow_pickle=False)

score_log = np.load('data/score_log.npy', allow_pickle=False)
pos_log = np.load('data/pos_log.npy', allow_pickle=False)
force_log = np.load('data/force_log.npy', allow_pickle=False)

# Generate graphs
plt.figure(1)
plt.plot(avg_train, label='AVG. Rewards')
plt.plot(sum_train, alpha=0.25, label='Sum. Rewards')
plt.grid(True)
plt.xlabel('Training Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.title('Agent Training Profile')
plt.savefig('data/Agent Training Profile.png')

plt.figure(2)
plt.plot(pos_log, label='Position')
plt.plot(force_log, alpha=0.5, label='Force')
plt.grid(True)
plt.xlabel('Testing Episodes')
plt.ylabel('Mean Units')
plt.legend(loc='best')
plt.title('Agent Testing Profile')
plt.savefig('data/Agent Testing Profile.png')
