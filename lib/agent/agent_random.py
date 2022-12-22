"""
agent_random.py
~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: SAT APR 25 2021.
~~~~~~~~~~~~~~~~~~~~~~
"""
import numpy as np

from lib.agent.basic_matrix_agent import BasicMatrixAgent


class RandomAgent(BasicMatrixAgent):

    def __init__(self, fov_scope):

        super(RandomAgent, self).__init__(fov_scope=fov_scope)

    def policy(self, local_env_matrix):

        action, action_probability = self.random_walk_in_single_agent_system_without_memory(local_env_matrix)

        return action

    def random_walk_in_single_agent_system_without_memory(self, local_env_matrix):
        """
        Open space:

            There is no other agents or obstacles in the neighborhood,
            except the the agent itself.

        Policy:

            Move to a random open one-step away position.

        """

        # 2d numpy array with the shape (x, 2), where 0 <= x <= 4.

        open_neighbors = self.get_open_axial_neighbors(self.local_own_position, local_env_matrix)

        if open_neighbors.shape[0] == 0:

            next_position = self.local_own_position.copy()
            action_probability = 1

        else:

            candidates = np.vstack((open_neighbors, self.local_own_position))
            n_candidates = len(candidates)

            # Random select one candidate.

            idx = np.random.choice(n_candidates, 1)[0]
            next_position = candidates[idx]
            action_probability = 1 / n_candidates

        # Get the action.
        # 1d numpy array with the shape(2,), np.array([delta_x, delta_y]).

        direction = next_position - self.local_own_position

        next_action = self.direction_action[tuple(direction)]

        return next_action, action_probability


def test():
    pass


if __name__ == "__main__":
    test()
