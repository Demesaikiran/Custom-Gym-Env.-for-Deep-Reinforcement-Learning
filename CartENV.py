# Library Imports
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class System(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.FPS": 50}

    def __init__(self):
        """
        Description,
            Initializes the openai-gym environment with it's features.
        """
        # Mass of the cart
        self.masscart = 1.0

        # Kinematics Solver
        self.tau = 0.02
        self.kinematics_integrator = "euler"

        # Set ENV. Constraints
        self.x_threshold = 2
        self.episode_limit = 0

        # Setting for Env. Verbose
        obs_limits = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        # Params. ENV. Spaces
        self.observation_space = spaces.Box(-obs_limits,
                                            obs_limits,
                                            dtype=np.float32)

        # Init. Renders & Environment
        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        """
        Description,
            Starts the ENV. at random place.

        Args:
            seed ([int], optional): Starts ENV. at same place if enabled. Defaults to None.

        Returns:
            [seed]: Resulting ENV. State seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Description,
            Computes the physics of cart based on action applied.

        Args:
            action ([np.float32]): Apply +ve or -ve force.

        Returns:
            ([np.array size=(2,1) type=np.float32]): Next State
            ([np.float32]): Reward as norm distance from '0' state
            ([np.bool: ENV]). Terminal condition.
        """
        # Build State as Position, Velocity (a = F/m)
        x, x_dot = self.state

        # Exert Force(=action) on Mass
        xacc = action / self.masscart

        # Solve Control System
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc

        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot

        # Revise the State
        self.state = (x, x_dot)

        # Engg. Terminal Condition Reward Position
        if x < -self.x_threshold or x > self.x_threshold:
            done = True
        elif self.episode_limit == 500:
            done = True
        else:
            done = False

        if not done:
            reward = -np.linalg.norm(0 - x)
            self.episode_limit += 1
            return np.array(self.state, dtype=np.float32), reward, False
        else:
            reward = -np.linalg.norm(0 - x)
            self.episode_limit += 1
            return np.array(self.state, dtype=np.float32), reward, True

    def reset(self):
        """
        Description,
            Resets the ENV.

        Returns:
            ([np.array size=(2,1) type=np.float32]): Random State
        """
        self.state = self.np_random.uniform(low=-1, high=1, size=(2,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        """
        Description,
            Renders the ENV.
        """
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0
        self.carttrans.set_translation(cartx, carty)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        """
        Description,
            Closes the rendering window if provoked.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
