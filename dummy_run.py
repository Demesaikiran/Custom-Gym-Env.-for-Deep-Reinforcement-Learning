# Library Imports
from CartENV import System
import numpy as np


# Init. ENV.
env = System()
env.reset()

# Render and Close
done = False
while not done:
    env.render()
    action = np.random.uniform(-10, 10)
    state, reward, done = env.step(action)
    print(f'State:{state} \t Action:{action:3.2f} \
          \t Reward:{reward:3.2f} \t Done:{done}')
env.close()
