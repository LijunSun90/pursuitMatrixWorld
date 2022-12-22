import numpy as np
from lib.environment.matrix_world import MatrixWorld as pursuit


env = pursuit()

env.reset(seed=1000)

for _ in range(50):

    rewards, observations, _, game_done, info = env.last(is_evader=False)

    actions = np.random.choice(5, 4)

    env.step_swarm(actions)

    env.render()

