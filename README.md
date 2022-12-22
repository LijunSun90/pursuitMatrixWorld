
# Multi-agent pursuit in matrix world (pursuitMW)

Multi-agent pursuit in matrix world (pursuitMW) is a partially observable Markov game (POMG) 
between a swarm of pursuers and a swarm of evaders.
Algorithms can be developed for the pursuers, evaders, or both of them.

## Task definition.

- Evader: red square.
- Pursuer: blue square.
  
- A capture: four pursuers surround one evader so that the evader cannot move.
- Game termination: all evaders are captured or the maximum allowed time is reached.

Configurable parameters:

- Grid world size.
- Partial observable scope of agent.
- No. of pursuers.
- No. of evaders.
- Reward function.

Observation space:

- Shape: (fov_scope, fov_scope, 3) with the three channels corresponds to evader, pursuer, and obstacle.
- Value: int, which indicates the number of entities of a specific type (evader, pursuer, obstacles).

Action space:

- Discrete: {0, 1, 2, 3, 4}.

Reward function:

- Capture an evader: 5.
- Neighbor an evader: 0.1
- Collision: -0.2.
- Every time step before termination: -0.05.


## Task challenges.

- Crowded multi-agent POMG environment: 
  The higher the swarm density of agents, the more challenging the multi-agent coordination is. 

- Sparse multi-agent POMG environment:
  The sparser the agents, the more challenging.
  For example, when the environment is very big, the evaders and pursuers are both several, 
  it is challenging for pursuers to cooperate to search and capture the targets,
  with only partial observation of the environment and no communication.


## API.

See example_basic_api.py.

    import numpy as np
    from lib.environment.matrix_world import MatrixWorld as pursuit

    env = pursuit()
    env.reset(seed=1000)
    for _ in range(100):
        rewards, observations, _, game_done, info = env.last(is_evader=False)
        actions = np.random.choice(5, 4)
        env.step_swarm(actions)
        env.render()


## Dependency.

- numpy.


## Cite paper:

This multi-agent pursuit environment is based on the pursuit environment used in the papers

- Sun, L., Lyu, C. and Shi, Y., 2020. 
  Cooperative coevolution of real predator robots and virtual robots in the pursuit domain. 
  Applied Soft Computing, 89, p.106098. (https://github.com/LijunSun90/pursuitCCPSOR)
  
- Sun, L., Lyu, C., Shi, Y. and Lin, C.T., 2021, June. 
  Multiple-Preys Pursuit based on Biquadratic Assignment Problem. 
  In 2021 IEEE Congress on Evolutionary Computation (CEC) (pp. 1585-1592). IEEE.

- Sun, L., Chang, Y.C., Lyu, C., Shi, Y., Shi, Y. and Lin, C.T., 2022. 
  Toward multi-target self-organizing pursuit in a partially observable Markov game. 
  arXiv preprint arXiv:2206.12330. (https://github.com/LijunSun90/pursuitFSC2)


If you use this environment code, please cite

    Sun, L., Chang, Y.C., Lyu, C., Shi, Y., Shi, Y. and Lin, C.T., 2022. 
    Toward multi-target self-organizing pursuit in a partially observable Markov game. 
    arXiv preprint arXiv:2206.12330. 



