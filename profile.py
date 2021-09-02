import numpy as np
import os
import matplotlib.pyplot as plt

# Init. path
path = os.getcwd()

# Load all the data frames
sum_train = np.load('data/score_history.npy', allow_pickle=False)
avg_train = np.load('data/avg_history.npy', allow_pickle=False)

# Generate graphs
plt.figure(1)
plt.plot(sum_train, alpha=0.25, label='Sum. Rewards')
plt.plot(avg_train, label='AVG. Rewards')
plt.grid(True)
plt.xlabel('Training Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.title('Agent Training Profile')
plt.savefig('data/Agent Training Profile.png')