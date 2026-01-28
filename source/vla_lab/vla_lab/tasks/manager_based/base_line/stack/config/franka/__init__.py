# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Stack-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_joint_pos_env_cfg:FrankaCubeStackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)


##
# Gr00t Dataset Generator
##

gym.register(
    id="VlaLab-BaseLine-Stack-IK-Rel-Gr00t-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_rel_gr00t_env_cfg:FrankaCubeStackGr00tEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="VlaLab-BaseLine-Stack-IK-Rel-Gr00t-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_rel_gr00t_env_cfg:FrankaCubeStackGr00tPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)


