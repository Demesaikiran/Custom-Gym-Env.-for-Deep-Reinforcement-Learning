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
    env.step(action)
env.close()