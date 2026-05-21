# 1. vec_env.py

from __future__ import annotations

import numpy as np


from env import ChineseCheckersEnv

import random 
ALL_AXES = [
    ["red", "blue"],
    ["yellow", "purple"], 
    ["lawn green", "gray0"]
]

class VecEnv:
    """
    Vectorized Chinese Checkers environments.

    IMPORTANT:
    PPO only sees learner turns.

    Opponent turns are automatically played internally.

    This preserves correct PPO semantics for
    turn-based multi-agent environments.
    """

    def __init__(self,
                 num_envs,
                 env_kwargs,
                 opponent_sampler):

        self.num_envs = num_envs

        self.envs = [
            ChineseCheckersEnv(**env_kwargs)
            for _ in range(num_envs)
        ]

        self.opponent_sampler = opponent_sampler

        self.states = []

        self.active_colors = []

        self.opponents = []

        self.dones = []

        # --------------------------------------------
        # Initialize all envs
        # --------------------------------------------

        for env in self.envs:

            (
                state,
                active_color,
                opps,
            ) = self._reset_env(env)

            self.states.append(state)

            self.active_colors.append(
                active_color
            )

            self.opponents.append(opps)

            self.dones.append(False)

    # ====================================================
    # RESET SINGLE ENV
    # ====================================================

    def _reset_env(self, env):

        state = env.reset(
            pins_advanced=0
        )

        active_color = np.random.choice(
            env.assigned
        )

        opps = {}

        for c in env.assigned:

            if c == active_color:
                continue

            opps[c] = self.opponent_sampler(
                env,
                c,
            )

        # --------------------------------------------
        # Auto-play until learner turn
        # --------------------------------------------

        state = self._advance_to_learner_turn(
            env,
            state,
            active_color,
            opps,
        )

        return (
            state,
            active_color,
            opps,
        )

    # ====================================================
    # ADVANCE OPPONENT TURNS
    # ====================================================

    def _advance_to_learner_turn(self,
                                 env,
                                 state,
                                 active_color,
                                 opponents):

        while (
            not env.done
            and env.current_color != active_color
        ):

            mask = env.build_action_mask(
                env.current_color
            )

            # ----------------------------------------
            # No legal moves
            # ----------------------------------------

            if mask.sum() == 0:

                env._advance_turn()

                continue

            opponent = opponents[
                env.current_color
            ]

            
            action = opponent.select_action(
                env,
                state,
                mask,
                None,
            )

            # Convert tensor -> python int
            if hasattr(action, "item"):
                action = int(action.item())
            else:
                action = int(action)
            


            state, _, _ = env.step(action)

        return state

    # ====================================================
    # GET STATES
    # ====================================================

    def get_states(self):

        return np.stack(
            self.states
        ).astype(np.float32)

    # ====================================================
    # GET MASKS
    # ====================================================

    def get_masks(self):

        masks = []

        for i, env in enumerate(self.envs):

            mask = env.build_action_mask(
                self.active_colors[i]
            )

            masks.append(mask)

        return np.stack(
            masks
        ).astype(np.float32)

    # ====================================================
    # STEP
    # ====================================================

    def step(self, actions):

        next_states = []

        rewards = []

        dones = []

        infos = []

        for i, env in enumerate(self.envs):

            action = int(actions[i])

            # ----------------------------------------------------
            # Env already finished
            # ----------------------------------------------------

            if env.done:

                (
                    state,
                    active_color,
                    opps,
                ) = self._reset_env(env)

                self.states[i] = state
                self.active_colors[i] = active_color
                self.opponents[i] = opps

                next_states.append(state)

                rewards.append(0.0)

                dones.append(True)

                infos.append(
                    {"reset": True}
                )

                continue

            # ----------------------------------------------------
            # Learner action
            # ----------------------------------------------------

            state, reward, done = env.step(
                action
            )

            # ----------------------------------------------------
            # Opponent auto-play
            # ----------------------------------------------------

            if not done:

                state = self._advance_to_learner_turn(
                    env,
                    state,
                    self.active_colors[i],
                    self.opponents[i],
                )

                # IMPORTANT:
                # env may become done during opponent turns
                done = env.done

            # ----------------------------------------------------
            # Store current state
            # ----------------------------------------------------

            self.states[i] = state

            next_states.append(state)

            rewards.append(reward)

            dones.append(done)

            infos.append({})

            # ----------------------------------------------------
            # Reset AFTER storing transition
            # ----------------------------------------------------

            if done:

                (
                    new_state,
                    active_color,
                    opps,
                ) = self._reset_env(env)

                self.states[i] = new_state
                self.active_colors[i] = active_color
                self.opponents[i] = opps

        return (
            np.stack(next_states).astype(
                np.float32
            ),
            np.array(
                rewards,
                dtype=np.float32,
            ),
            np.array(
                dones,
                dtype=np.float32,
            ),
            infos,
        )


