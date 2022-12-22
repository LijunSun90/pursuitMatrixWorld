"""
matrix_world.py
~~~~~~~~~~~~~~~~~~~

AUTHOR: LIJUN SUN.
DATE: MON 27 APR 2020.
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import os
import copy
import numpy as np
import shutil

import matplotlib.pyplot as plt


class MatrixWorld:

    # Encoding.
    # Coordinate origin: upper left.
    # Clockwise.
    # 0     1  2  3  4  5   6   7   8
    # still N  E  S  W  NE  SE  SW  NW
    action_direction = {0: (0, 0),
                        1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1),
                        5: (-1, 1), 6: (1, 1), 7: (1, -1), 8: (-1, -1)}
    direction_action = dict([(value, key) for key, value in action_direction.items()])
    actions_orthogonal = list(range(5))
    actions_diagonal = list(range(9))

    @classmethod
    def create_padded_env_matrix_from_vectors(cls, world_rows, world_columns,
                                              fov_scope, evaders, pursuers,
                                              obstacles):
        """
        :param world_rows: int.
        :param world_columns: int.
        :param fov_scope: int. An odd number.
        :param evaders: 2d numpy array of shape (x, 2) where 0 <= x.
        :param pursuers: 2d numpy array of shape (x, 2) where 0 <= x.
        :param obstacles: 2d numpy array of shape (x, 2) where 0 <= x.
        :return: 3d numpy array of shape (world_rows + fov_scope - 1,
                                          world_columns + fov_scope - 1, 4)
            channel 0: the evaders matrix,
            channel 1: the pursuers matrix,
            channel 2: is the obstacles matrix.
            channel 3: unknown map.
            In a channel, the pixel value is 1 in an agent's location, else 0.
        """

        # Parameters.
        fov_radius = int(0.5 * (fov_scope - 1))
        fov_offsets_in_padded = np.array([fov_radius] * 2)
        fov_pad_width = np.array([fov_radius] * 2)
        # [lower_bound, upper_bound).
        fov_mask_in_padded = np.array([[-fov_radius] * 2, [fov_radius + 1] * 2]) + fov_offsets_in_padded

        # Create matrix.
        padded_env_matrix = np.zeros((world_rows + fov_scope - 1,
                                      world_columns + fov_scope - 1, 4), dtype=int)

        for channel in [0, 1, 2, 3]:
            if channel == 2:
                # Obstacles matrix are padded with 1, borders are obstacles.
                padded_value = 1
            else:
                padded_value = 0

            if channel == 3:
                # Unknown map matrix is initially all 1s.
                env_matrix_channel = np.ones((world_rows, world_columns), dtype=int)
            else:
                env_matrix_channel = np.zeros((world_rows, world_columns), dtype=int)

            padded_env_matrix[:, :, channel] = \
                np.pad(env_matrix_channel,
                       pad_width=((fov_pad_width[0], fov_pad_width[1]),
                                  (fov_pad_width[0], fov_pad_width[1])),
                       mode="constant",
                       constant_values=(padded_value, padded_value))

        # Write data.
        positions_in_padded = evaders + fov_offsets_in_padded
        padded_env_matrix[positions_in_padded[:, 0], positions_in_padded[:, 1], 0] = 1

        positions_in_padded = pursuers + fov_offsets_in_padded
        padded_env_matrix[positions_in_padded[:, 0], positions_in_padded[:, 1], 1] = 1

        positions_in_padded = obstacles + fov_offsets_in_padded
        padded_env_matrix[positions_in_padded[:, 0], positions_in_padded[:, 1], 2] = 1

        padded_env_matrix[:, :, 3] = cls.update_env_matrix_unknown_map(fov_mask_in_padded,
                                                                       padded_env_matrix[:, :, 3],
                                                                       pursuers)

        return copy.deepcopy(padded_env_matrix)

    @classmethod
    def update_env_matrix_unknown_map(cls, fov_mask_in_padded, padded_env_matrix_unknown_map, pursuers):
        """
        :param fov_mask_in_padded: 2d numpy array of shape (2, 2),
            which is [[row_min, column_min], [row_max, column_max]].
        :param padded_env_matrix_unknown_map: 2d numpy array of shape
            (world_rows + fov_scope - 1, world_columns + fov_scope - 1).
        :param pursuers: 2d numpy array of shape (x, 2) or
            1d numpy array of shape (2,).
        :return: 2d numpy array of the same shape of
            `padded_env_matrix_unknown_map`.

        Mark the local perceptible scope of a pursuer as known region.
        """
        # 1d to 2d array.
        if len(pursuers.shape) == 1:
            pursuers = pursuers.reshape((1, -1))

        for pursuer in pursuers:
            fov_idx = pursuer + fov_mask_in_padded
            padded_env_matrix_unknown_map[fov_idx[0, 0]: fov_idx[1, 0],
                                          fov_idx[0, 1]: fov_idx[1, 1]] = 0

        return copy.deepcopy(padded_env_matrix_unknown_map)

    @classmethod
    def get_inf_norm_distance(cls, from_position, to_position):
        """
        :param from_position: 1d numpy array with the shape(2,).
        :param to_position: 1d numpy array with the shape(2,).
        :return: int, the inf-norm.
        """
        delta = to_position - from_position
        distance = np.linalg.norm(delta, ord=np.inf).astype(int)

        return distance.copy()

    @classmethod
    def get_distance(cls, from_position, to_position):
        """
        :param from_position: 1d numpy array with the shape(2,).
        :param to_position: 1d numpy array with the shape(2,).
        :return: int, the 1-norm.

        Manhattan distance or City distance.
        """
        delta = to_position - from_position
        distance = np.linalg.norm(delta, ord=1).astype(int)

        return distance.copy()

    @classmethod
    def render_a_matrix(cls, env_matrix, is_display=True, is_fixed_size=False, grid_on=True, tick_labels_on=False):

        # 1. Render.

        ax_image = plt.imshow(env_matrix, origin="upper")

        # 2. Set figure parameters.

        # add the colorbar using the figure's method,
        # telling which mappable we're talking about and
        # which axes object it should be near
        ax_image.figure.colorbar(ax_image, ax=ax_image.axes)

        ax_image.figure.set_frameon(True)
        if is_fixed_size:
            ax_image.figure.set_figwidth(10)
            ax_image.figure.set_figheight(10)

        # 3. Set axes parameters.

        x_ticks = np.arange(-0.5, env_matrix.shape[1], 1)
        y_ticks = np.arange(-0.5, env_matrix.shape[0], 1)

        ax_image.axes.set_xticks(x_ticks)
        ax_image.axes.set_yticks(y_ticks)
        if not tick_labels_on:
            ax_image.axes.set_xticklabels([])
            ax_image.axes.set_yticklabels([])
        else:
            ax_image.axes.set_xticklabels(x_ticks, rotation=90)

        ax_image.axes.tick_params(which='both', direction='in',
                                  left=False, bottom=False,
                                  right=False, top=False)

        ax_image.axes.margins(0, 0)

        ax_image.axes.grid(grid_on)

        # 4. Control the display.

        if is_display:
            # plt.pause(interval)
            plt.show()

        # 5.
        plt.close()

    def __init__(self,
                 world_rows=6, world_columns=6,
                 n_evaders=1, n_pursuers=4,
                 fov_scope=11,
                 max_env_cycles=500,
                 diagonal_move=False,
                 obstacle_density=0,
                 save_path="data/frames"):
        """
        :param world_rows: int, corresponds to the 1st axis.
        :param world_columns: int, corresponds to the 2nd axis.
        :param n_evaders: int, >= 0.
        :param n_pursuers: int, >= 0.
        :param fov_scope: int, >=1, an odd integer.
            The scope of the field of view of agents.
            The agent locates in the center of its own local field of view.
        :param obstacle_density: float.
        :param save_path: string.
        """

        # Initialization parameters.

        self.world_rows = world_rows
        self.world_columns = world_columns
        self.world_scope = np.array([self.world_rows, self.world_columns])

        self.n_evaders = n_evaders
        self.n_pursuers = n_pursuers

        self.max_env_cycles = max_env_cycles

        self.n_actions = 9 if diagonal_move else 5

        # Reward parameters.

        self.reward_collision = -0.2

        self.reward_pursuer_capture = 5

        self.reward_pursuer_neighboring_evader = 0.1

        self.reward_pursuer_non_terminal = -0.05
        self.reward_evader_non_terminal = 0.05

        # Env status: share parameters between functions.

        self.collision_with_obstacle_status_evaders = np.array([False] * self.n_evaders)
        self.collision_with_obstacle_status_pursuers = np.array([False] * self.n_pursuers)
        self.collision_with_swarm_agent_status_pursuers = np.array([False] * self.n_pursuers)

        self.n_collision_with_boundaries_pursuers = 0

        self.n_collision_with_obstacle_evaders = 0
        self.n_collision_with_obstacle_pursuers = 0
        self.n_collided_pursuers = 0
        self.n_multiagent_collision_events_pursuers = 0

        # Indicate whether the game is over.

        self.done = False

        self.frame_no = 0

        # 2-D numpy array,
        # where each row is a 2-D point in the global coordinate system.
        # Shape: (n_evaders, 2)
        self.evaders = None
        # Shape: (n_pursuers, 2)
        self.pursuers = None
        # Shape: (None, 2)
        self.obstacles = None

        self.env_matrix = None
        self.padded_env_matrix = None

        # FOV parameters.

        self.fov_scope = fov_scope
        self.fov_radius = int(0.5 * (self.fov_scope - 1))

        self.fov_offsets_in_padded = np.array([self.fov_radius] * 2)

        # [lower_bound, upper_bound).
        self.fov_mask_in_padded = \
            np.array([[-self.fov_radius] * 2, [self.fov_radius + 1] * 2]) + self.fov_offsets_in_padded

        self.fov_global_scope_in_padded = \
            np.array([[0, 0], [self.world_rows, self.world_columns]]) + self.fov_offsets_in_padded

        # Neighbors.

        # N, E, S, W.
        self.axial_neighbors_mask = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        self.two_steps_away_neighbors_mask = \
            np.array([[-2, 0], [0, 2], [2, 0], [0, -2],
                      [-1, 1], [1, 1], [1, -1], [-1, -1]])

        self.obstacle_density = obstacle_density

        self.save_path = save_path

        # Processed variables.

        self.n_cells = self.world_rows * self.world_columns
        self.n_obstacles = round(self.n_cells * self.obstacle_density)

        # Get coordinates of the whole world.
        # 0, 1, ..., (world_rows - 1); 0, 1, ..., (world_columns - 1).
        # For example,
        # array([[[0, 0, 0],
        #         [1, 1, 1],
        #         [2, 2, 2]],
        #        [[0, 1, 2],
        #         [0, 1, 2],
        #         [0, 1, 2]]])
        self.meshgrid_x, self.meshgrid_y = np.mgrid[0:self.world_rows, 0:self.world_columns]

        # Example of meshgrid[0].flatten() is
        # array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        # Example of meshgrid[1].flatten() is
        # array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.xs, self.ys = self.meshgrid_x.flatten(), self.meshgrid_y.flatten()

        # Rendering parameters.
        self.title_template = 'Step = %d'

        self.x_ticks = np.arange(-0.5, self.world_columns, 1)
        self.y_ticks = np.arange(-0.5, self.world_rows, 1)

        # Saving parameters.
        self.frame_prefix = "MatrixWorld"

    def set_frame_prefix(self, frame_prefix):
        """
        :param frame_prefix: str
        """
        self.frame_prefix = frame_prefix

    def reset_env_status(self):

        self.frame_no = 0
        self.done = False

        # Share parameters between functions.

        self.collision_with_obstacle_status_evaders = np.array([False] * self.n_evaders)
        self.collision_with_obstacle_status_pursuers = np.array([False] * self.n_pursuers)
        self.collision_with_swarm_agent_status_pursuers = np.array([False] * self.n_pursuers)

        self.n_collision_with_boundaries_pursuers = 0

        self.n_collision_with_obstacle_evaders = 0
        self.n_collision_with_obstacle_pursuers = 0
        self.n_collided_pursuers = 0
        self.n_multiagent_collision_events_pursuers = 0

    def reset(self, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # 0. Reset env status parameters.
        self.reset_env_status()

        # 1. Create obstacles.
        empty_cells_index = np.arange(self.n_cells).tolist()

        self.obstacles, empty_cells_index = self.random_select(empty_cells_index.copy(), self.n_obstacles)

        # 2. Create evaders.
        self.evaders, empty_cells_index = self.random_select(empty_cells_index.copy(), self.n_evaders)

        # 3. Create pursuers.
        self.pursuers, empty_cells_index = self.random_select(empty_cells_index.copy(), self.n_pursuers)

        # Matrix representation.
        # shape: (world_rows, world_columns, 4)
        # channel 0: the evaders matrix,
        # channel 1: the pursuers matrix,
        # channel 2: is the obstacles matrix.
        # channel 3: unknown map.
        # In each channel, the pixel value is 1 in an agent's location, else 0.
        self.padded_env_matrix = self.create_padded_env_matrix_from_vectors(
            self.world_rows, self.world_columns, self.fov_scope,
            self.evaders, self.pursuers, self.obstacles)

        self.env_matrix = self.padded_env_matrix[
                          self.fov_global_scope_in_padded[0, 0]:
                          self.fov_global_scope_in_padded[1, 0],
                          self.fov_global_scope_in_padded[0, 1]:
                          self.fov_global_scope_in_padded[1, 1], :]

        # Restore the random.
        # if seed is not None:
        #     np.random.seed()

        # Return.

        _, observations, _, game_done, _ = self.last(is_evader=False)

        return observations, game_done

    def random_select(self, empty_cells_index, n_select):
        """
        Random select ``n_select`` cells out of the total cells
        ``empty_cells_index``.

        :param empty_cells_index: a list of integers where each integer
                                  corresponds to some kind of index.
        :param n_select: int, >=0.
        :return: (entities, empty_cells_index), where ``entities`` is a
                 (n_select, 2) numpy array, and ``empty_cells_index`` is a
                 (n - n_select, 2) numpy array.
        """

        # Indexes.

        idx_entities = np.random.choice(empty_cells_index, n_select, replace=False)

        # Maintain the left empty cells.

        for idx in idx_entities:
            empty_cells_index.remove(idx)

        # Coordinates.
        # Indexes in 2D space.
        #       |       |     |
        # 0 -idx:0--idx:1--idx:2-
        # 1 -idx:3--idx:4--idx:5-
        # 2 -idx:6--idx:7--idx:8-
        #       |       |     |
        #       0       1     2
        xs_entities, ys_entities = self.xs[idx_entities], self.ys[idx_entities]

        # Get the entities positions.

        entities = np.vstack((xs_entities, ys_entities)).T

        return copy.deepcopy(entities), copy.deepcopy(empty_cells_index)

    def reset_with_parameters(self, evaders, pursuers):

        # 0. Reset env status parameters.

        self.reset_env_status()

        # 1. Create obstacles.

        empty_cells_index = np.arange(self.n_cells).tolist()
        self.obstacles, empty_cells_index = self.random_select(empty_cells_index.copy(), self.n_obstacles)

        self.evaders = evaders
        self.pursuers = pursuers

        # Matrix representation.
        # shape: (world_rows, world_columns, 4)
        # channel 0: the evaders matrix,
        # channel 1: the pursuers matrix,
        # channel 2: is the obstacles matrix.
        # channel 3: unknown map.
        # In each channel, the pixel value is 1 in an agent's location, else 0.

        self.padded_env_matrix = self.create_padded_env_matrix_from_vectors(
            self.world_rows, self.world_columns, self.fov_scope,
            self.evaders, self.pursuers, self.obstacles)

        self.env_matrix = self.padded_env_matrix[
                          self.fov_global_scope_in_padded[0, 0]:
                          self.fov_global_scope_in_padded[1, 0],
                          self.fov_global_scope_in_padded[0, 1]:
                          self.fov_global_scope_in_padded[1, 1], :]

    def update_a_evader(self, idx_evader, new_position):
        """
        :param idx_evader: int, >=0.
        :param new_position: 1d numpy array of shape (2,).
        :return: None.
        """

        # 1. Update vector.

        old_position = self.get_a_evader(idx_evader)
        self.evaders[idx_evader, :] = new_position

        # 2. Update the channel.

        old_position_in_padded = old_position + self.fov_offsets_in_padded
        new_position_in_padded = new_position + self.fov_offsets_in_padded

        self.padded_env_matrix[old_position_in_padded[0],
                               old_position_in_padded[1], 0] -= 1
        self.padded_env_matrix[new_position_in_padded[0],
                               new_position_in_padded[1], 0] += 1

    def update_a_pursuer(self, idx_pursuer, new_position):
        """
        :param idx_pursuer: int, >=0.
        :param new_position: 1d numpy array of shape (2,).
        :return: None.
        """

        # 1. Update vector.

        old_position = self.get_a_pursuer(idx_pursuer)
        self.pursuers[idx_pursuer, :] = new_position

        # 2. Update unknown map.

        fov_idx = self.fov_mask_in_padded + new_position
        self.padded_env_matrix[fov_idx[0, 0]: fov_idx[1, 0],
                               fov_idx[0, 1]: fov_idx[1, 1], 3] = 0

        # 3. Update pursuer channel.

        old_position_in_padded = old_position + self.fov_offsets_in_padded
        new_position_in_padded = new_position + self.fov_offsets_in_padded

        self.padded_env_matrix[old_position_in_padded[0],
                               old_position_in_padded[1], 1] -= 1
        self.padded_env_matrix[new_position_in_padded[0],
                               new_position_in_padded[1], 1] += 1

    def perceive_swarm(self, is_evader=False):

        observations = []

        if is_evader:

            for idx_evader in range(self.n_evaders):

                observations.append(self.perceive(idx_evader, is_evader=True)[1])

        else:

            for idx_pursuer in range(self.n_pursuers):

                observations.append(self.perceive(idx_pursuer, is_evader=False)[1])

        observations = np.stack(observations)

        env_vectors = dict()
        env_vectors["all_evaders"] = self.get_all_evaders()
        env_vectors["all_pursuers"] = self.get_all_pursuers()

        # return observations, env_time
        return observations, env_vectors

    def perceive(self, idx_agent, is_evader=False, remove_current_agent=True):
        """
        :param idx_agent: int, >= 0.
        :param is_evader: boolean.
        :param remove_current_agent:
        :return: a tuple, (own_position, local_env_matrix),
            "own_position" is 1d numpy array of shape (2,).
            "local_matrix" is 3d numpy array of shape
                (self.fov_scope, self.fov_scope, 3)
                with each channel being
                (local_evader, local_pursuers, local_obstacles).
        """

        if is_evader:
            # 1d numpy array with the shape (2,).
            global_position = self.get_a_evader(idx_agent)
        else:
            # 1d numpy array with the shape (2,).
            global_position = self.get_a_pursuer(idx_agent)

        fov_idx = self.fov_mask_in_padded + global_position

        local_evaders = \
            self.padded_env_matrix[fov_idx[0, 0]: fov_idx[1, 0],
                                   fov_idx[0, 1]: fov_idx[1, 1], 0].copy()
        local_pursuers = \
            self.padded_env_matrix[fov_idx[0, 0]: fov_idx[1, 0],
                                   fov_idx[0, 1]: fov_idx[1, 1], 1].copy()
        local_obstacles = \
            self.padded_env_matrix[fov_idx[0, 0]: fov_idx[1, 0],
                                   fov_idx[0, 1]: fov_idx[1, 1], 2].copy()

        if remove_current_agent:

            if is_evader:

                local_evaders[self.fov_offsets_in_padded[0],
                              self.fov_offsets_in_padded[1]] -= 1

            else:

                local_pursuers[self.fov_offsets_in_padded[0],
                               self.fov_offsets_in_padded[1]] -= 1

        local_env_matrix = np.stack((local_evaders, local_pursuers, local_obstacles), axis=2)

        return global_position.copy(), local_env_matrix.copy()

    def perceive_matrix_globally(self, idx_agent, is_evader=False):

        if is_evader:
            # 1d numpy array with the shape (2,).
            center_position = self.get_a_evader(idx_agent)
        else:
            # 1d numpy array with the shape (2,).
            center_position = self.get_a_pursuer(idx_agent)

        return center_position.copy(), self.env_matrix[:, :, [0, 1, 2]].copy()

    def perceive_globally(self, idx_agent, is_evader=False):
        """
        :param idx_agent: int, >= 0.
        :param is_evader: boolean.
        :return: a dict.
        """

        if is_evader:
            # 1d numpy array with the shape (2,).
            center_position = self.get_a_evader(idx_agent)
        else:
            # 1d numpy array with the shape (2,).
            center_position = self.get_a_pursuer(idx_agent)

        env_vectors = dict()
        env_vectors["own_position"] = center_position
        env_vectors["all_evaders"] = self.get_all_evaders()
        env_vectors["all_pursuers"] = self.get_all_pursuers()
        env_vectors["obstacles"] = self.get_all_obstacles()

        return copy.deepcopy(env_vectors)

    def get_a_evader(self, idx_evader=0):
        """
        :param idx_evader:
        :return: 1d numpy array with the shape (2,).
        """
        return self.evaders[idx_evader, :].copy()

    def get_a_pursuer(self, idx_pursuer):
        """
        :param idx_pursuer:
        :return: 1d numpy array with the shape (2,).
        """
        return self.pursuers[idx_pursuer, :].copy()

    def get_all_evaders(self):

        return self.evaders.copy()

    def get_all_pursuers(self):

        return self.pursuers.copy()

    def get_all_obstacles(self):
        return self.obstacles.copy()

    def last(self, is_evader=False):

        last_reward = self.get_reward(is_evader=is_evader)

        # current_observation, env_time = self.perceive_swarm(is_evader=is_evader)

        current_observation, env_vectors = self.perceive_swarm(is_evader=is_evader)

        # current_game_status

        all_captured, capture_rate = self.is_all_captured()

        if is_evader:

            global_own_positions = env_vectors["all_evaders"]
            info = self.n_collision_with_obstacle_evaders

        else:

            global_own_positions = env_vectors["all_pursuers"]

            info = capture_rate, \
                self.n_collision_with_obstacle_pursuers, \
                self.n_multiagent_collision_events_pursuers, \
                self.n_collided_pursuers

        return last_reward, current_observation, global_own_positions, all_captured, info

    def get_reward(self, is_evader=False):

        if is_evader:

            reward_time = self.reward_evader_non_terminal

            reward_evader = [0] * self.n_evaders

            for idx_evader in range(self.n_evaders):

                n_collisions = self.is_collide(self.get_a_evader(idx_evader), exclude_this_position_value=True)
                reward_evader_collision = self.reward_collision * n_collisions
                reward_evader[idx_evader] = reward_evader_collision + reward_time

            return reward_evader

        else:

            reward_pursuer_capture = self.get_reward_capture()

            reward_pursuer_neighbor_evader = self.get_reward_neighbor_evader()

            reward_time = self.reward_pursuer_non_terminal

            reward_pursuer = [0] * self.n_pursuers

            for idx_pursuer in range(self.n_pursuers):

                reward_pursuer_collision = \
                    self.reward_collision * (self.collision_with_obstacle_status_pursuers[idx_pursuer] or
                                             self.collision_with_swarm_agent_status_pursuers[idx_pursuer])

                reward_pursuer[idx_pursuer] = \
                    reward_pursuer_collision + \
                    reward_pursuer_capture[idx_pursuer] + \
                    reward_pursuer_neighbor_evader[idx_pursuer] * (reward_pursuer_capture[idx_pursuer] == 0) + \
                    reward_time

            return reward_pursuer

    def get_reward_capture(self):

        reward_pursuer = [0] * self.n_pursuers

        pursuer_idx = dict()
        for idx_pursuer in range(self.n_pursuers):
            position = self.get_a_pursuer(idx_pursuer)
            if tuple(position) in pursuer_idx.keys():
                pursuer_idx[tuple(position)].append(idx_pursuer)
            else:
                pursuer_idx[tuple(position)] = [idx_pursuer]

        for idx_evader in range(self.n_evaders):

            evader_position = self.get_a_evader(idx_evader)

            if not self.is_captured(evader_position):
                continue

            for mask in self.axial_neighbors_mask:

                neighbor_position = evader_position + mask + self.fov_offsets_in_padded
                neighbor_pursuer_existence = \
                    self.padded_env_matrix[neighbor_position[0], neighbor_position[1], 1]

                if neighbor_pursuer_existence > 0:

                    idx_pursuer = pursuer_idx[tuple(neighbor_position - self.fov_offsets_in_padded)]

                    for idx in idx_pursuer:
                        reward_pursuer[idx] = self.reward_pursuer_capture

        return reward_pursuer

    def get_reward_neighbor_evader(self):

        reward_pursuer = np.zeros(self.n_pursuers)

        for idx_pursuer in range(self.n_pursuers):

            position = self.get_a_pursuer(idx_pursuer)

            axial_neighbors = self.axial_neighbors_mask + position + self.fov_offsets_in_padded

            n_axial_neighbors = self.padded_env_matrix[axial_neighbors[:, 0], axial_neighbors[:, 1], 0].sum()

            if n_axial_neighbors > 0:
                reward_pursuer[idx_pursuer] = self.reward_pursuer_neighboring_evader

        return reward_pursuer

    def step_swarm(self, actions, is_evader=False, allow_inter_swarm_collision=True):

        n_agents = self.n_evaders if is_evader else self.n_pursuers

        collision_with_boundary_status = np.array([False] * n_agents)

        collision_with_obstacle_status = np.array([False] * n_agents)

        collision_with_swarm_agent_status_pursuers = np.array([False] * n_agents)

        position_multiagent_collision_status = dict()

        # Prepare variables.

        from_positions = []
        desired_positions = []

        for idx_agent, action in enumerate(actions.tolist()):

            from_position = self.get_a_evader(idx_agent) if is_evader else self.get_a_pursuer(idx_agent)

            desired_position = self.move_to(from_position, action)

            # 0. Collide with boundary.

            if self.is_out_of_boundary(desired_position):
                collision_with_boundary_status[idx_agent] = True

            # 1. Collide with static obstacle.
            # (agents in the other swarm is considered as a sort of environment obstacle,
            # if not allow_inter_swarm_collision.)
            # The desired position is invalid if it collide with a static obstacle.

            if self.is_collide_with_obstacle(desired_position, is_evader=is_evader,
                                             allow_inter_swarm_collision=allow_inter_swarm_collision):

                # Bounce back to original position and get reward punishment.

                desired_position = from_position

                collision_with_obstacle_status[idx_agent] = True

            from_positions.append(tuple(from_position))
            desired_positions.append(tuple(desired_position))

            # 1.1 Collide with agents in the other swarm.

            if allow_inter_swarm_collision and self.is_collide_with_other_swarm_agents(desired_position,
                                                                                       is_evader=is_evader):

                collision_with_obstacle_status[idx_agent] = True

        # 2. Collide with other agents of the same type, i.e., collisions within the swarm.

        to_positions = np.array(desired_positions)

        for idx_agent, desired_pos in enumerate(desired_positions):

            # Whether the desired position collide with other agent's desired position.

            if desired_positions.count(desired_pos) > 1:

                position_multiagent_collision_status[desired_pos] = True

                collision_with_swarm_agent_status_pursuers[idx_agent] = True

        # Return information.

        if is_evader:

            self.collision_with_obstacle_status_evaders = collision_with_obstacle_status
            self.n_collision_with_obstacle_evaders = sum(collision_with_obstacle_status)

        else:

            self.collision_with_obstacle_status_pursuers = collision_with_obstacle_status
            self.collision_with_swarm_agent_status_pursuers = collision_with_swarm_agent_status_pursuers

            self.n_collision_with_boundaries_pursuers = sum(collision_with_boundary_status)
            self.n_collision_with_obstacle_pursuers = sum(collision_with_obstacle_status)
            self.n_collided_pursuers = sum(collision_with_swarm_agent_status_pursuers)
            self.n_multiagent_collision_events_pursuers = len(position_multiagent_collision_status)

        # Swarm move.

        for idx_agent, to_position in enumerate(to_positions):

            if is_evader:

                self.update_a_evader(idx_agent, to_position)

            else:

                self.update_a_pursuer(idx_agent, to_position)

    def step(self, idx_agent, action, is_evader=False):
        """
        :param idx_agent: index of a pursuer or a evader.
        :param action: int, 0 ~ 5 or 0 ~ 9 depending on ``self.move_diagonal``.
        :param is_evader: if False, move the pursuer;
                        if True, move the evader.
        :return: a tuple, (executable, collide) where both are boolean,
            indicate whether the action is executable or not, and
            indicate whether there is a collision.
            Change the position of the ``idx_agent`` if it is valid.
        """

        # Shape: (2, )
        from_position = self.get_a_evader(idx_agent) if is_evader else self.get_a_pursuer(idx_agent)

        to_position = self.move_to(from_position, action)

        # Check validation.
        # Include the execution status of keeping still.
        if to_position.tolist() == from_position.tolist():
            return

        if self.is_out_of_boundary(to_position):
            return

        # Change the position.
        if is_evader:
            self.update_a_evader(idx_agent, to_position)
        else:
            self.update_a_pursuer(idx_agent, to_position)

    def move_to(self, from_position, action):
        """
        :param from_position: 1d numpy array with the shape: (2, ).
        :param action: int, 0 ~ 5 or 0 ~ 9 depending on ``self.move_diagonal``.
        :return: 1d numpy array with the shape: (2, ).

        The position if the ``action`` is performed, regardless of its
        validation.
        """
        direction = self.action_direction[action]
        to_position = from_position + direction

        return to_position.copy()

    def is_out_of_boundary(self, position):
        yes_or_no = False
        if (position < [0, 0]).any() or (position >= [self.world_rows, self.world_columns]).any():
            yes_or_no = True

        return yes_or_no

    def is_collide_with_obstacle(self, new_position, is_evader=False, allow_inter_swarm_collision=True):
        # channel: 0-evader, 1-pursuer, 2-obstacle, 3-unknown region.
        if allow_inter_swarm_collision:
            idx_obstacle_channel = [2]
        else:
            idx_obstacle_channel = [1, 2] if is_evader else [0, 2]

        new_position = new_position + self.fov_offsets_in_padded
        pixel_values_in_new_position = \
            self.padded_env_matrix[new_position[0], new_position[1], idx_obstacle_channel].sum()

        collided = False if pixel_values_in_new_position == 0 else True

        return collided

    def is_collide_with_other_swarm_agents(self, new_position, is_evader=False):
        # channel: 0-evader, 1-pursuer, 2-obstacle, 3-unknown region.
        idx_other_swarm_channel = 1 if is_evader else 0

        new_position = new_position + self.fov_offsets_in_padded
        pixel_values_in_new_position = \
            self.padded_env_matrix[new_position[0], new_position[1], idx_other_swarm_channel].sum()

        collided = False if pixel_values_in_new_position == 0 else True

        return collided

    def is_collide(self, new_position, exclude_this_position_value=False):
        """
        Check the whether ``new_position`` collide with others in the global
        scope.

        ``new_position`` is valid
        if it additionally does not locate out the grid world boundaries.
        If it move out of the boundaries, it can also been seen that the agent
        collides with the boundaries, and so also a kind of collision.

        :param new_position: 1d numpy array with the shape (2,).
        :param exclude_this_position_value:
        :return: boolean, indicates  valid or not.
        """
        new_position = new_position + self.fov_offsets_in_padded

        pixel_values_in_new_position = self.padded_env_matrix[new_position[0], new_position[1], :-1].sum()

        if exclude_this_position_value:

            n_collisions = max(0, pixel_values_in_new_position - 1)

        else:

            n_collisions = pixel_values_in_new_position

        return n_collisions

    def is_captured(self, evader_position):

        capture_positions = self.axial_neighbors_mask + evader_position + self.fov_offsets_in_padded

        occupied_capture_positions = self.padded_env_matrix[capture_positions[:, 0],
                                                            capture_positions[:, 1], :-1].sum(axis=1)

        # Valid only if collision is not allowed in the space.
        # Otherwise, more than one agents can occupy the same position.

        captured = 1 if (occupied_capture_positions > 0).all() else 0

        return captured

    def is_all_captured(self):

        all_evaders = self.get_all_evaders()

        n_captured = 0

        for evader in all_evaders:

            n_captured += self.is_captured(evader)

        capture_rate = n_captured / len(all_evaders)

        # return yes_no
        return n_captured == self.n_evaders, capture_rate

    def get_n_collision_events(self):

        # Channel: 0: evader. 1: pursuer. 2: obstacle.
        env_matrix = self.env_matrix.copy()

        # Each position should have at most one entity if there is no collisions.
        env_matrix = env_matrix.sum(axis=2)
        env_matrix = np.maximum(0, env_matrix - 1)

        # Only count the collision events, not the number of collided agents.
        env_matrix = np.minimum(1, env_matrix)

        n_collision_events = env_matrix.sum()

        return n_collision_events

    def get_scope_mask_in_env_matrix(self, circle_center, radius):
        """
        :param circle_center: 1d numpy array of shape (2,).
        :param radius: int, > 0.
        :return: 2d numpy array of shape (2, 2),
            which is [[row_min, column_min], [row_max, column_max]]
        """

        scope_mask = np.array([[-radius] * 2, [radius] * 2]) + circle_center

        index_min = np.maximum([0, 0], scope_mask[0, :])
        index_max = np.minimum([self.world_rows, self.world_columns],
                               scope_mask[1, :])

        scope_mask = np.vstack((index_min, index_max))

        return copy.deepcopy(scope_mask)

    def render(self, is_display=True, interval=0.0001,
               is_save=False, is_fixed_size=False,
               grid_on=True, tick_labels_on=False,
               show_pursuer_idx=False, show_evader_idx=False,
               show_frame_title=True,
               use_input_env_matrix=False,
               env_matrix=None,
               save_name=None,
               clear_dir=True):

        # 0. Prepare the directory.

        if self.frame_no == 0 and is_save:
            self.create_directory(clear_dir=clear_dir)

        # White: 255, 255, 255.
        # Black: 0, 0, 0.
        # Yellow, 255, 255, 0.
        # Silver: 192, 192, 192

        # Background: white.
        # evader: red. pursuer: blue. Obstacle: black. Unknown regions: yellow.

        #               R    G    B
        # Background: 255, 255, 255
        # evader:       255,   0,   0
        # pursuer:     0,   0, 255
        # Obstacle:     0,   0,   0
        # Unknown:    255, 255,   0
        # Pursuer fov:255, 255,   0

        # 1. Prepare data.

        if not use_input_env_matrix:
            env_matrix = self.env_matrix.copy()
            x_ticks = self.x_ticks
            y_ticks = self.y_ticks
            all_pursuers = self.get_all_pursuers()
            all_evaders = self.get_all_evaders()
        else:
            x_ticks = np.arange(-0.5, env_matrix.shape[1], 1)
            y_ticks = np.arange(-0.5, env_matrix.shape[0], 1)
            all_pursuers = np.stack(np.where(np.where(env_matrix[:, :, 1]) == 1), axis=1)
            all_evaders = np.stack(np.where(np.where(env_matrix[:, :, 0]) == 1), axis=1)

        # White world.
        rgb_env = np.ones((env_matrix.shape[0], env_matrix.shape[1], 3))

        # Fov scope of pursuers: green, 255, 255, 0.
        # 0 -> 1, 0 -> 1.
        # rgb_env[:, :, [0, 2]] = \
        #     np.logical_and(rgb_env[:, :, [0, 2]],
        #                    env_matrix[:, :, 3].reshape(env_matrix.shape[0], env_matrix.shape[1], 1))

        # for i_row in range(env_matrix.shape[0]):
        #     for j_column in range(env_matrix.shape[1]):
        #         if env_matrix[i_row, j_column, 0] == 1:
        #             rgb_env[i_row, j_column, :] = [1, 0, 0]
        #         if env_matrix[i_row, j_column, 1] == 1:
        #             rgb_env[i_row, j_column, :] = [0, 0, 1]
        #         if env_matrix[i_row, j_column, 2] == 1:
        #             rgb_env[i_row, j_column, :] = [0, 0, 0]

        # Obstacle: black, 0, 0, 0.
        # 1 -> 0.
        rgb_env = np.logical_xor(rgb_env, np.expand_dims(env_matrix[:, :, 2], axis=2))

        # evader: red, 255, 0, 0.
        # 1 -> 0.
        rgb_env[:, :, [1, 2]] = np.logical_xor(rgb_env[:, :, [1, 2]], np.expand_dims(env_matrix[:, :, 0], axis=2))

        # pursuer: blue, 0, 0, 255.
        # 1 -> 0.
        rgb_env[:, :, [0, 1]] = np.logical_xor(rgb_env[:, :, [0, 1]], np.expand_dims(env_matrix[:, :, 1], axis=2))

        # Unknown map: green, 255, 255, 0.
        # 1 -> 0, 0 -> 0.
        # rgb_env[:, :, 2] = np.logical_and(rgb_env[:, :, 2], 1 - env_matrix[:, :, 3])

        # Fov scope of pursuers: green, 255, 255, 0.
        # 0 -> 1, 0 -> 1.
        # rgb_env[:, :, 2] = np.logical_and(rgb_env[:, :, 2], env_matrix[:, :, 3])

        rgb_env = rgb_env * 255

        # 2. Render.

        ax_image = plt.imshow(rgb_env, origin="upper")

        if show_pursuer_idx:
            for idx, pursuer in enumerate(all_pursuers):
                text = plt.text(pursuer[1], pursuer[0], str(idx))

        if show_evader_idx:
            for idx, evader in enumerate(all_evaders):
                text = plt.text(evader[1], evader[0], str(idx))

        # 3. Set figure parameters.

        ax_image.figure.set_frameon(True)
        if is_fixed_size:
            ax_image.figure.set_figwidth(10)
            ax_image.figure.set_figheight(10)

        # 4. Set axes parameters.

        ax_image.axes.set_xticks(x_ticks)
        ax_image.axes.set_yticks(y_ticks)
        if not tick_labels_on:
            ax_image.axes.set_xticklabels([])
            ax_image.axes.set_yticklabels([])
        else:
            ax_image.axes.set_xticklabels(x_ticks, rotation=90)

        ax_image.axes.tick_params(which='both', direction='in',
                                  left=False, bottom=False,
                                  right=False, top=False)

        ax_image.axes.margins(0, 0)

        ax_image.axes.grid(grid_on)

        # 5. Set title.

        if show_frame_title:
            plt.title(self.title_template % self.frame_no)

        # 6. Control the display.

        if is_display:
            plt.pause(interval)
            # plt.show()
            # pass

        if is_save:
            if save_name is None:
                plt.savefig(os.path.join(self.save_path,
                                         self.frame_prefix + "{0:0=4d}".format(self.frame_no)),
                            bbox_inches='tight')

                # plt.imsave(self.save_path + self.frame_prefix + "{0:0=4d}".
                #            format(self.frame_no) + ".png",
                #            arr=rgb_env, format="png")
            else:
                plt.savefig(os.path.join(self.save_path, save_name), bbox_inches='tight')

        # 7. Update.

        self.frame_no += 1

        # 8.
        # plt.close()

    def create_directory(self, clear_dir=True):
        if os.path.exists(self.save_path):
            if clear_dir:
                shutil.rmtree(self.save_path)
                os.makedirs(self.save_path)
        else:
            os.makedirs(self.save_path)


def test():
    print("Testing ...")

    world_rows = 40
    world_columns = 40

    n_evaders = 4
    # n_pursuers = 4 * (n_evaders + 1)
    n_pursuers = 4 * n_evaders

    env = MatrixWorld(world_rows, world_columns,
                      n_evaders=n_evaders, n_pursuers=n_pursuers)

    env.reset(seed=3)

    print("Step 0...")
    env.render(is_display=True, interval=0.5,
               is_save=True, is_fixed_size=False,
               grid_on=True, tick_labels_on=False,
               show_pursuer_idx=True,
               show_evader_idx=True,
               show_frame_title=False)

    # test_perceive(env, idx_pursuer=12)

    print("Step 1...")

    env.render(is_display=True, interval=0.5,
               is_save=True, is_fixed_size=False,
               grid_on=True, tick_labels_on=True,
               show_pursuer_idx=True,
               show_evader_idx=True)



if __name__ == "__main__":
    test()
    # test_reward()
    pass
