# Library Imports
from CartENV import System
from TD3 import Agent
import numpy as np
import argparse

# Init. ENV.
env = System()
obs = env.reset()

# Init. the Agent
agent = Agent(alpha=0.001, beta=0.001, tau=0.005,
              env=env, batch_size=128)


# Defintion for training the untrained agent
def train(n_games):
    """
    Description,
        Trains the agent for n_games.

    Args:
        n_games ([int]): No. of games to train the agent.
    """
    done = False

    # Init. & Start Training
    score_history = []
    avg_history = []
    best_score = env.reward_range[0]
    avg_score = 0

    for i in range(n_games):
        score = 0
        done = False

        # Initial Reset of Environment
        observation = env.reset()
        for j in range(env.max_steps):
            action = agent.choose_action(observation)
            observation_, reward, done, = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            score += reward
            if done:
                break

        # Optimize the Agent
        agent.optimize()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            print(f'Episode:{i} \t ACC. Rewards: {score:3.2f} \t \
                AVG. Rewards: {avg_score:3.2f} \t *** MODEL SAVED! ***')
        else:
            print(f'Episode:{i} \t ACC. Rewards: {score:3.2f} \t \
                AVG. Rewards: {avg_score:3.2f}')

        # Save the Training data and Model Loss
        np.save('data/score_history', score_history, allow_pickle=False)
        np.save('data/avg_history', avg_history, allow_pickle=False)


# Main Section
if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", default=2500)
    args = parser.parse_args()
    n_games = int(args.n_games)

    # Run the test
    print(f'Training Agent for: {n_games} games.')
    train(n_games)
