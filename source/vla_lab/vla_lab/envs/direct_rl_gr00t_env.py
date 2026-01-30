# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import abstractmethod
from typing import Any, Dict

import omni.kit.app

from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg
from isaaclab.envs.direct_rl_env import DirectRLEnv

from .utils import AsyncGr00tInferenceClient


class DirectRLGr00tEnv(DirectRLEnv):

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.gr00t_chunk_size = 16
        self.gr00t_policy = AsyncGr00tInferenceClient(host="localhost", port=5555)  # TODO: edit port id
        print("Initialize Gr00t Client Node")

        self.gr00t_actions: Dict[str, Any] | None = None
        self.processed_gr00t_actions: torch.Tensor | None = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Resets all the environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset all the environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        # Call Initial Gr00t Action
        gr00t_observations: Dict[str, Any] = self._get_gr00t_observations()
        self.gr00t_actions = self.gr00t_policy.get_action_sync(gr00t_observations)

        # return observations
        return self._get_observations(), self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # if self.episode_length_buf[0].item() == 0:
        #     self.stime = time.time()

        if self.episode_length_buf[0] % self.gr00t_chunk_size == 0:

            if self.episode_length_buf[0].item() != 0:  # First Step after reset
                try:
                    self.gr00t_actions = self.gr00t_policy.get_result()
                except Exception as e:
                    print(f"Error getting Gr00t action: {e}")
                    self.gr00t_actions = torch.zeros((self.num_envs, self.gr00t_chunk_size, 6), device=self.device)

            groot_actions_pos = torch.tensor(self.gr00t_actions["action.eef_position_delta"], dtype=torch.float32, device=self.device)
            groot_actions_rot = torch.tensor(self.gr00t_actions["action.eef_rotation_delta"], dtype=torch.float32, device=self.device)
            groot_actions_delta_pose = torch.cat(
                (groot_actions_pos, groot_actions_rot), dim=-1
            )

            self.processed_gr00t_actions = groot_actions_delta_pose # [64, 16, 6]

            # forge env contain predicition of sucess, so we expand gr00t actions to match the num envs
            if self.action_space.shape[-1] != self.processed_gr00t_actions.shape[-1]:
                self.processed_gr00t_actions = torch.cat(
                    (self.processed_gr00t_actions, torch.zeros((self.num_envs, self.gr00t_chunk_size, 1), device=self.device)), dim=-1
                )

        elif self.episode_length_buf[0] % self.gr00t_chunk_size == 8:
            
            self.gr00t_policy.request_action(self._get_gr00t_observations())

            
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # process actions
        # self._pre_physics_step(action)
        # self._pre_physics_step(self.processed_gr00t_actions[:, self.episode_length_buf[0] % self.gr00t_chunk_size, :])
        self._pre_physics_step(action + self.processed_gr00t_actions[:, self.episode_length_buf[0] % self.gr00t_chunk_size, :])

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # if self.episode_length_buf[0].item() == self.max_episode_length - 2:
        #     self.etime = time.time()
        #     print(f"one episode time: {self.etime - self.stime} sec")

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    @abstractmethod
    def _get_gr00t_observations(self) -> Dict[str, Any]:
        """Compute and return the observations for the environment.

        Returns:
            The observations for the environment.
        """
        raise NotImplementedError(f"Please implement the '_get_gr00t_observations' method for {self.__class__.__name__}.")