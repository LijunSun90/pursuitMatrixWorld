"""
basic_matrix_agent.py
~~~~~~~~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: WED JUN 2 2021.
"""
import numpy as np


class BasicMatrixAgent:

    # Encoding.
    # Clockwise.
    # 0     1  2  3  4  5   6   7   8
    # still N  E  S  W  NE  SE  SW  NW

    action_direction = {0: (0, 0),
                        1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1),
                        5: (-1, 1), 6: (1, 1), 7: (1, -1), 8: (-1, -1)}

    direction_action = \
        dict([(value, key) for key, value in action_direction.items()])

    # N, E, S, W.
    axial_neighbors_mask = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

    @classmethod
    def get_open_axial_neighbors(cls, local_position_concerned, local_env_matrix):
        """
        :param local_position_concerned: 1d numpy with the shape (2,).
        :param local_env_matrix: 3d numpy array with the shape (m, n, 3).
        :return: 2d numpy array with the shape (x, 2), where 0 <= x <= 4,
                 depending on how many axial neighbors are still open,
                 i.e., not be occupied.
        """

        neighbors = cls.axial_neighbors_mask + local_position_concerned

        open_idx = []

        for idx, neighbor in enumerate(neighbors):

            if not cls.is_collide(neighbor, local_env_matrix):

                open_idx.append(idx)

        open_neighbors = neighbors[open_idx, :]

        return open_neighbors.copy()

    @classmethod
    def is_collide(cls, new_position, local_env_matrix, exclude_this_position_value=False):

        pixel_values_in_new_position = local_env_matrix[new_position[0], new_position[1], :-1].sum()

        if exclude_this_position_value:

            n_collisions = max(0, pixel_values_in_new_position - 1)

        else:

            n_collisions = pixel_values_in_new_position

        return n_collisions

    def __init__(self, fov_scope):

        self.fov_scope = fov_scope

        self.fov_radius = int((fov_scope - 1) / 2)

        self.local_own_position = np.array([self.fov_radius] * 2)

        # 2d numpy array of shape (4, 2).
        self.axial_neighbors_mask = self.axial_neighbors_mask

    def update_local_env_vectors(self, local_env_matrix):
        """
        local_env_matrix -> local_env_vectors.
        """

        # Parse.
        local_evaders_matrix = local_env_matrix[:, :, 0].copy()
        local_pursuers_matrix = local_env_matrix[:, :, 1].copy()
        local_obstacles_matrix = local_env_matrix[:, :, 2].copy()

        # Vector representation.

        # 2d numpy array of shape (x, 2) where 0 <= x.
        local_evaders = np.asarray(np.where(local_evaders_matrix > 0)).T
        local_pursuers = np.asarray(np.where(local_pursuers_matrix > 0)).T
        local_obstacles = np.asarray(np.where(local_obstacles_matrix > 0)).T

        # Remove local captured evaders.

        # for evader in local_evaders:
        #     self.is_captured(evader, local_env_matrix)

        local_env_vectors = {"local_evaders": local_evaders,
                             "local_pursuers": local_pursuers,
                             "local_obstacles": local_obstacles}

        return local_env_vectors

    def is_captured(self, evader_position, local_env_matrix):

        capture_positions = evader_position + self.axial_neighbors_mask

        valid_index = []
        for idx, position in enumerate(capture_positions):
            if (position >= 0).all() and (position < self.fov_scope).all():
                valid_index.append(idx)

        capture_positions = capture_positions[valid_index, :]

        occupied_capture_positions = local_env_matrix[capture_positions[:, 0],
                                                      capture_positions[:, 1], :].sum(axis=1)

        # Valid only if collision is not allowed in the space.
        # Otherwise, more than one agents can occupy the same position.
        yes_no = True if (occupied_capture_positions > 0).all()else False

        return yes_no


def test():
    pass


if __name__ == "__main__":
    test()
