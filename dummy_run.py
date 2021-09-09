# Library Imports
from CartENV import System
import numpy as np
import argparse

# Init. ENV.
env = System()
env.reset()


# Defintion to Simulate the environment
def simulate(mode):
    """
    Description,
        Simulates environment with random actions.

    Args:
        mode ([bool]): If True, renders the environment.
    """
    done = False
    render = mode

    for i in range(env.max_steps):
        # Render if Enabled
        if render:
            env.render()
        else:
            pass

        action = np.random.uniform(-10, 10)
        state, reward, done = env.step(action)

        if (i % 10 == 0):
            print(f'Step: {i} \
                  \n\tState:{np.around(state, 4)} \n\tAction:{action:3.2f} \
                  \n\tReward:{reward:3.2f} \n\tDone:{done}\n')

        if done:
            break

    if render:
        env.close()


# Main Section
if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", help='Enable/ Disable Render')
    args = parser.parse_args()
    if args.render == 'True':
        mode = True
    else:
        mode = False

    # Run the test
    print(f'Enabled Rendering Mode: {mode}')
    simulate(mode)
