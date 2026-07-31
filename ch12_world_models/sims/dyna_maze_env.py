# Sutton blocking maze. Chapter: Forecasting, Dreaming and Learning.
# 6x9 deterministic gridworld with a phase-switchable wall, used to
# demonstrate Dyna-Q's planning amplification and the Dyna-Q+ extension's
# recovery from environment change.

import numpy as np


class BlockingMaze:
    """Sutton & Barto Ch 8 blocking maze.

    Layout (row 0 at top, row 5 at bottom; '.' = open, 'W' = wall):

      Phase 1                     Phase 2
      .......G                    .......G
      ........                    ........
      WWWWWWWW. <- wall row       .WWWWWWWW
      ........                    ........
      ........                    ........
      ...S....                    ...S....

    The wall is on row 2 in both phases; only the open column moves
    (col 8 in Phase 1, col 0 in Phase 2). The flip happens automatically
    when env.step() returns its phase-switch time step.

    State: (row, col) ; Action: 0=up, 1=right, 2=down, 3=left.
    Reward: 0 on every step, +1 on goal arrival; episode terminates on goal.
    Transitions are deterministic; stepping into a wall or off the grid
    leaves the agent in place.
    """

    N_ROWS = 6
    N_COLS = 9
    START = (5, 3)
    GOAL = (0, 8)
    ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left

    PHASE1_WALL_COLS = (0, 1, 2, 3, 4, 5, 6, 7)
    PHASE2_WALL_COLS = (1, 2, 3, 4, 5, 6, 7, 8)
    WALL_ROW = 2

    def __init__(self, t_switch=1000, t_total=3000, episode_cap=200):
        self.t_switch = t_switch
        self.t_total = t_total
        self.episode_cap = episode_cap
        self.t_global = 0
        self.t_episode = 0
        self.pos = self.START

    def reset(self):
        self.pos = self.START
        self.t_episode = 0
        return self.pos

    def reset_global(self):
        self.t_global = 0
        self.t_episode = 0
        self.pos = self.START

    def _is_wall(self, r, c):
        if r < 0 or r >= self.N_ROWS or c < 0 or c >= self.N_COLS:
            return True
        if r != self.WALL_ROW:
            return False
        wall_cols = (self.PHASE1_WALL_COLS if self.t_global < self.t_switch
                     else self.PHASE2_WALL_COLS)
        return c in wall_cols

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        r, c = self.pos
        nr, nc = r + dr, c + dc
        if self._is_wall(nr, nc):
            nr, nc = r, c  # no-op
        self.pos = (nr, nc)
        self.t_global += 1
        self.t_episode += 1
        if (nr, nc) == self.GOAL:
            return self.pos, 1.0, True, {'phase': 1 if self.t_global <= self.t_switch else 2}
        if self.t_episode >= self.episode_cap:
            return self.pos, 0.0, True, {'phase': 1 if self.t_global <= self.t_switch else 2,
                                          'truncated': True}
        return self.pos, 0.0, False, {'phase': 1 if self.t_global <= self.t_switch else 2}

    def state_id(self, pos=None):
        """Linearize (r, c) into a single int in [0, N_ROWS * N_COLS)."""
        r, c = pos if pos is not None else self.pos
        return r * self.N_COLS + c

    def n_states(self):
        return self.N_ROWS * self.N_COLS

    def n_actions(self):
        return 4


if __name__ == "__main__":
    env = BlockingMaze(t_switch=1000, t_total=3000)
    print(f"Maze: {env.N_ROWS} x {env.N_COLS}, start={env.START}, goal={env.GOAL}")
    print(f"Phase 1 wall cols: {env.PHASE1_WALL_COLS}")
    print(f"Phase 2 wall cols: {env.PHASE2_WALL_COLS}")

    # Smoke test: optimal-ish path through Phase 1 (up-right).
    env.reset_global()
    env.reset()
    rng = np.random.default_rng(0)
    steps = 0
    rewards = 0
    while env.t_global < 50:
        a = int(rng.integers(0, 4))
        s, r, done, info = env.step(a)
        rewards += r
        steps += 1
        if done:
            env.reset()
    print(f"\nRandom 50-step rollout: {rewards} reward, {steps} steps total, phase={info.get('phase')}")
    print(f"Final pos: {env.pos}")
