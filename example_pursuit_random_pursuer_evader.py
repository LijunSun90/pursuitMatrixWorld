import argparse
import numpy as np
# import torch

from lib.environment.matrix_world import MatrixWorld as pursuit
from lib.agent.agent_random import RandomAgent


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", "-s", type=int, default=0)

    # Environment.

    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--render_save", type=bool, default=False)

    parser.add_argument("--max_episode_length", type=int, default=50)
    parser.add_argument("--world_rows", type=int, default=80)
    parser.add_argument("--world_columns", type=int, default=80)
    parser.add_argument("--fov_scope", type=int, default=11)
    parser.add_argument("--n_pursuers", type=int, default=256)
    parser.add_argument("--n_evaders", type=int, default=64)

    return parser.parse_args()


def train(arg_list):

    # 1. Initialization.

    env = pursuit(world_rows=arg_list.world_rows, world_columns=arg_list.world_columns,
                  n_evaders=arg_list.n_evaders, n_pursuers=arg_list.n_pursuers,
                  fov_scope=11, max_env_cycles=arg_list.max_episode_length)

    seed = arg_list.seed + 10000
    # torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)

    pursuer = RandomAgent(fov_scope=arg_list.fov_scope)
    evader = RandomAgent(fov_scope=arg_list.fov_scope)

    episode_return = 0
    episode_length = 0
    episode_capture_rate = 0
    episode_n_collisions_with_obstacles = 0
    episode_multiagent_collisions_events = 0
    episode_n_collided_agents = 0

    # One episode.

    for t in range(arg_list.max_episode_length):

        # Render.

        if arg_list.render:

            env.render(is_display=True, is_save=arg_list.render_save,
                       is_fixed_size=False, grid_on=True, tick_labels_on=False,
                       show_pursuer_idx=False, show_evader_idx=False)

        # Pursuers.

        # Swarm of pursuers simultaneously 1. observe, 2. make decision, 3 act.

        _, observations, global_own_positions, game_done, info = env.last(is_evader=False)

        actions = []

        for i_pursuer in range(arg_list.n_pursuers):
            action = pursuer.policy(observations[i_pursuer])
            actions.append(action)

        actions = np.array(actions)

        if not game_done:

            env.step_swarm(actions, is_evader=False)

        rewards, _, _, game_done, (capture_rate,
                                   n_collision_with_obstacle,
                                   n_multiagent_collision_events,
                                   n_collided_pursuers) = env.last(is_evader=False)

        # Log.

        episode_return += np.mean(rewards)
        episode_length += 1
        episode_capture_rate = capture_rate
        episode_n_collisions_with_obstacles += n_collision_with_obstacle
        episode_multiagent_collisions_events += n_multiagent_collision_events
        episode_n_collided_agents += n_collided_pursuers

        # Evaders.

        # To ensure safe movement of evaders, we let them sequentially observe, make decision, and act,
        # since the evaders are taken as part of the environment in this experiment.

        for i_evader in range(arg_list.n_evaders):

            if game_done:
                continue

            _, observation_evader = env.perceive(idx_agent=i_evader, is_evader=True)
            action_evader = evader.policy(observation_evader)
            env.step(i_evader, action_evader, is_evader=True)

        episode_timeout = (episode_length == arg_list.max_episode_length)
        episode_terminal = game_done or episode_timeout

        if episode_terminal:

            print("Episode, "
                  "Return: {:4f}, "
                  "Time steps: {:4f}, "
                  "Capture rate: {:4f}, "
                  "Collisions with obstacles: {:4f}, "
                  "Multiagent collisions: {:4f}, "
                  "Collided agents: {:4f}".format(
                episode_return,
                episode_length,
                episode_capture_rate,
                episode_n_collisions_with_obstacles,
                episode_multiagent_collisions_events,
                episode_n_collided_agents))

            break

        pass


if __name__ == "__main__":

    print("Partially observable game (POMG): Random pursuers pursue random evaders. \n"
          "- Many collisions with obstacles and agents: "
          "Pursuers make random decisions and execute random actions simultaneously "
          "without considering possible multi-agent collisions.\n"
          "- Low capture rate: surrounding-based capture is hard for random agents.")

    arg_list = parse_args()

    train(arg_list)

    print('DONE!')