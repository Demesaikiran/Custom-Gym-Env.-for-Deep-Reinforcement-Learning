# Library Imports
from CartENV import System
from TD3 import Agent
import numpy as np
import argparse

# Init. ENV.
env = System()
obs = env.reset()
done = False

# Init. the Agent
agent = Agent(alpha=0.001, beta=0.001, tau=0.005,
              env=env, batch_size=128)

# Load Trained Weights
agent.actor.load_checkpoint()


# Defintion for Trained Agent Testing
def test(games, render):
    """
    Descrption,
        Test's the trained agent in the environement.

    Args:
        games ([int]): No. of games to be tested.
        render ([bool]): If enabled. Renders the enabled.
    """
    # Init. & Start Training
    n_games = games
    score_history = np.zeros(env.max_steps)
    force_history = np.zeros(env.max_steps)
    position_history = np.zeros(env.max_steps)

    for i in range(n_games):
        rewards = 0

        # Initial Reset of Environment
        observation = env.reset()
        for j in range(env.max_steps):
            # Render the ENV.
            if render:
                env.render()

            action = agent.choose_action(observation)
            observation_, reward, done, = env.step(action)
            observation = observation_
            rewards += reward
            score_history[j] += reward
            force_history[j] += action
            position_history[j] += observation[0]

        print(f'Done Profiling Test Episode: {i} \
            with Accumulated Rewards: {rewards}')

    # Mean the Logs.
    score_history /= env.max_steps
    force_history /= env.max_steps
    position_history /= env.max_steps

    # Save the Training data and Model Loss
    np.save('data/score_log', score_history, allow_pickle=False)
    np.save('data/force_log', force_history, allow_pickle=False)
    np.save('data/pos_log', position_history, allow_pickle=False)

    # Close the ENV.
    if render:
        env.close()


# Main Section
if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default=10)
    parser.add_argument("--render")
    args = parser.parse_args()
    games = int(args.games)
    if args.render == 'True':
        mode = True
    else:
        mode = False

    # Run the test
    print(f'Testing Agent for: {games} games.')
    test(games, mode)
